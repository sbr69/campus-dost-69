"""
User management router for organization team management.

MULTI-TENANCY: Superusers can manage users within their organization only.
Provides CRUD operations for team member management.
"""
import time
import re
import secrets
from datetime import datetime, timezone
from fastapi import APIRouter, Request, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from ..config import settings, logger
from ..dependencies import require_superuser, require_read_access, UserContext
from ..utils.limiter import limiter
from ..utils.validators import validate_no_null_bytes
from ..providers.database.firestore_init import get_db
from ..providers.database.activity import activity_provider

router = APIRouter(prefix="/api/v1/users", tags=["users"])


# --- Request/Response Models ---

class CreateUserRequest(BaseModel):
    """Request model for creating a new user."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = Field(None, max_length=200)
    role: str = Field("admin", pattern="^(admin|viewer|analyser)$")


class UpdateUserRequest(BaseModel):
    """Request model for updating a user."""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=200)
    role: Optional[str] = Field(None, pattern="^(admin|viewer|analyser)$")
    status: Optional[str] = Field(None, pattern="^(active|disabled)$")


class ResetPasswordRequest(BaseModel):
    """Request model for resetting a user's password."""
    new_password: str = Field(..., min_length=8)


class UserResponse(BaseModel):
    """User response model."""
    uid: str
    username: str
    email: str
    full_name: Optional[str] = None
    role: str
    org_id: str
    status: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# --- Helper Functions ---

def generate_uid(org_id: str, username: str) -> str:
    """Generate user ID in format: {org_id}_{username}"""
    return f"{org_id}_{username.lower()}"


def validate_username(username: str) -> bool:
    """Validate username format (alphanumeric, underscores, hyphens only)."""
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', username))


# --- Endpoints ---

@router.get("")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def list_users(
    request: Request,
    user: UserContext = Depends(require_superuser)
):
    """
    List all users in the organization.
    
    MULTI-TENANCY: Only returns users belonging to the superuser's organization.
    Requires superuser role.
    """
    start_time = time.perf_counter()
    db = get_db()
    
    try:
        # Query users in the organization
        users_ref = db.collection(settings.USERS_COLLECTION)
        query = users_ref.where("org_id", "==", user.org_id)
        docs = await query.get()
        
        users = []
        for doc in docs:
            data = doc.to_dict()
            users.append({
                "uid": doc.id,
                "username": data.get("username", doc.id),
                "email": data.get("email", ""),
                "full_name": data.get("full_name"),
                "role": data.get("role", "viewer"),
                "org_id": data.get("org_id", user.org_id),
                "status": data.get("status", "active"),
                "created_at": data.get("created_at").isoformat() if data.get("created_at") else None,
                "updated_at": data.get("updated_at").isoformat() if data.get("updated_at") else None
            })
        
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"Listed {len(users)} users for org={user.org_id} | {elapsed:.1f}ms")
        
        return {
            "status": "success",
            "users": users,
            "total": len(users)
        }
        
    except Exception as e:
        logger.error(f"Failed to list users for org={user.org_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to list users")


@router.get("/{user_id}")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def get_user(
    request: Request,
    user_id: str,
    user: UserContext = Depends(require_superuser)
):
    """
    Get a specific user by ID.
    
    MULTI-TENANCY: Only returns user if they belong to the superuser's organization.
    """
    start_time = time.perf_counter()
    
    validate_no_null_bytes(user_id, "user_id")
    if len(user_id) > 100:
        raise HTTPException(status_code=400, detail="Invalid user ID")
    
    db = get_db()
    
    try:
        doc = await db.collection(settings.USERS_COLLECTION).document(user_id).get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail="User not found")
        
        data = doc.to_dict()
        
        # Verify user belongs to same organization
        if data.get("org_id") != user.org_id:
            raise HTTPException(status_code=404, detail="User not found")
        
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"Retrieved user: {user_id} org={user.org_id} | {elapsed:.1f}ms")
        
        return {
            "status": "success",
            "user": {
                "uid": doc.id,
                "username": data.get("username", doc.id),
                "email": data.get("email", ""),
                "full_name": data.get("full_name"),
                "role": data.get("role", "viewer"),
                "org_id": data.get("org_id"),
                "status": data.get("status", "active"),
                "created_at": data.get("created_at").isoformat() if data.get("created_at") else None,
                "updated_at": data.get("updated_at").isoformat() if data.get("updated_at") else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user")


@router.post("")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def create_user(
    request: Request,
    background_tasks: BackgroundTasks,
    user: UserContext = Depends(require_superuser)
):
    """
    Create a new user in the organization.
    
    MULTI-TENANCY: User is created within the superuser's organization.
    Creates both Firebase Auth user and Firestore user document.
    Requires superuser role.
    
    Note: Sub-users cannot have superuser role (only org creators are superusers).
    """
    start_time = time.perf_counter()
    
    try:
        from ..services.firebase_auth import _init_firebase_admin
        from firebase_admin import auth as firebase_auth
    except ImportError:
        raise HTTPException(status_code=500, detail="Firebase auth not configured")
    
    req = await request.json()
    username = req.get("username", "").strip()
    email = req.get("email", "").strip()
    password = req.get("password", "")
    full_name = req.get("full_name", "").strip()
    role = req.get("role", "admin")
    
    # Validate inputs
    validate_no_null_bytes(username, "username")
    validate_no_null_bytes(email, "email")
    validate_no_null_bytes(password, "password")
    validate_no_null_bytes(full_name, "full_name")
    
    if not username or len(username) < 3 or len(username) > 50:
        raise HTTPException(status_code=400, detail="Username must be 3-50 characters")
    if not validate_username(username):
        raise HTTPException(status_code=400, detail="Username can only contain letters, numbers, hyphens, and underscores")
    if not email or len(email) > 254:
        raise HTTPException(status_code=400, detail="Invalid email address")
    if not password or len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if role not in ("admin", "viewer", "analyser"):
        raise HTTPException(status_code=400, detail="Invalid role. Must be admin, viewer, or analyser")
    
    db = get_db()
    
    # Generate user ID: {org_id}_{username}
    uid = generate_uid(user.org_id, username)
    
    try:
        # Check if user already exists
        existing_doc = await db.collection(settings.USERS_COLLECTION).document(uid).get()
        if existing_doc.exists:
            raise HTTPException(status_code=409, detail="Username already taken in this organization")
        
        # Initialize Firebase Admin
        _init_firebase_admin()
        
        # Create Firebase Auth user
        try:
            firebase_user = firebase_auth.create_user(
                uid=uid,
                email=email,
                password=password,
                display_name=full_name or username,
                email_verified=False
            )
            logger.info(f"Created Firebase user: {uid} ({email})")
        except Exception as e:
            error_msg = str(e)
            if "EMAIL_EXISTS" in error_msg or "already exists" in error_msg.lower():
                raise HTTPException(status_code=409, detail="Email already registered")
            elif "UID_ALREADY_EXISTS" in error_msg:
                raise HTTPException(status_code=409, detail="Username already taken")
            else:
                logger.error(f"Firebase user creation failed: {e}")
                raise HTTPException(status_code=500, detail=f"User creation failed")
        
        # Create user document in Firestore
        now = datetime.now(timezone.utc)
        user_data = {
            "uid": uid,
            "username": username,
            "email": email,
            "full_name": full_name or None,
            "org_id": user.org_id,
            "role": role,
            "status": "active",
            "created_at": now,
            "created_by": user.uid
        }
        await db.collection(settings.USERS_COLLECTION).document(uid).set(user_data)
        logger.info(f"Created user document: {uid}")
        
        # Log activity
        background_tasks.add_task(
            activity_provider.log_activity,
            user.org_id,
            "user_created",
            user.uid,
            "user",
            uid,
            {"username": username, "role": role}
        )
        
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"User created: {uid} org={user.org_id} | {elapsed:.1f}ms")
        
        return {
            "status": "success",
            "message": "User created successfully",
            "user": {
                "uid": uid,
                "username": username,
                "email": email,
                "full_name": full_name or None,
                "role": role,
                "org_id": user.org_id,
                "status": "active"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user")


@router.put("/{user_id}")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def update_user(
    request: Request,
    user_id: str,
    background_tasks: BackgroundTasks,
    user: UserContext = Depends(require_superuser)
):
    """
    Update a user's information.
    
    MULTI-TENANCY: Only updates user if they belong to the superuser's organization.
    Cannot change superuser role or demote superusers.
    """
    start_time = time.perf_counter()
    
    validate_no_null_bytes(user_id, "user_id")
    if len(user_id) > 100:
        raise HTTPException(status_code=400, detail="Invalid user ID")
    
    # Prevent self-modification of role/status
    if user_id == user.uid:
        raise HTTPException(status_code=400, detail="Cannot modify your own account through this endpoint")
    
    req = await request.json()
    db = get_db()
    
    try:
        # Get existing user
        doc = await db.collection(settings.USERS_COLLECTION).document(user_id).get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail="User not found")
        
        existing_data = doc.to_dict()
        
        # Verify user belongs to same organization
        if existing_data.get("org_id") != user.org_id:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Cannot modify superusers
        if existing_data.get("role") == "superuser":
            raise HTTPException(status_code=403, detail="Cannot modify superuser accounts")
        
        # Build update data
        update_data = {"updated_at": datetime.now(timezone.utc)}
        changes = []
        
        if "full_name" in req and req["full_name"] is not None:
            full_name = req["full_name"].strip()
            validate_no_null_bytes(full_name, "full_name")
            if len(full_name) > 200:
                raise HTTPException(status_code=400, detail="Full name too long")
            update_data["full_name"] = full_name or None
            changes.append("full_name")
        
        if "role" in req and req["role"] is not None:
            role = req["role"]
            if role not in ("admin", "viewer", "analyser"):
                raise HTTPException(status_code=400, detail="Invalid role")
            update_data["role"] = role
            changes.append("role")
        
        if "status" in req and req["status"] is not None:
            status = req["status"]
            if status not in ("active", "disabled"):
                raise HTTPException(status_code=400, detail="Invalid status")
            update_data["status"] = status
            changes.append("status")
            
            # If disabling, also disable in Firebase Auth
            if status == "disabled":
                try:
                    from ..services.firebase_auth import _init_firebase_admin
                    from firebase_admin import auth as firebase_auth
                    _init_firebase_admin()
                    firebase_auth.update_user(user_id, disabled=True)
                    logger.info(f"Disabled Firebase user: {user_id}")
                except Exception as e:
                    logger.warning(f"Failed to disable Firebase user: {e}")
            elif status == "active":
                try:
                    from ..services.firebase_auth import _init_firebase_admin
                    from firebase_admin import auth as firebase_auth
                    _init_firebase_admin()
                    firebase_auth.update_user(user_id, disabled=False)
                    logger.info(f"Enabled Firebase user: {user_id}")
                except Exception as e:
                    logger.warning(f"Failed to enable Firebase user: {e}")
        
        if not changes:
            return {
                "status": "success",
                "message": "No changes to update"
            }
        
        # Update Firestore document
        await db.collection(settings.USERS_COLLECTION).document(user_id).update(update_data)
        
        # Log activity
        background_tasks.add_task(
            activity_provider.log_activity,
            user.org_id,
            "user_updated",
            user.uid,
            "user",
            user_id,
            {"changes": changes}
        )
        
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"User updated: {user_id} changes={changes} | {elapsed:.1f}ms")
        
        return {
            "status": "success",
            "message": "User updated successfully",
            "changes": changes
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update user")


@router.delete("/{user_id}")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def delete_user(
    request: Request,
    user_id: str,
    background_tasks: BackgroundTasks,
    user: UserContext = Depends(require_superuser)
):
    """
    Delete a user from the organization.
    
    MULTI-TENANCY: Only deletes user if they belong to the superuser's organization.
    Deletes both Firebase Auth user and Firestore document.
    Cannot delete superuser accounts.
    """
    start_time = time.perf_counter()
    
    validate_no_null_bytes(user_id, "user_id")
    if len(user_id) > 100:
        raise HTTPException(status_code=400, detail="Invalid user ID")
    
    # Prevent self-deletion
    if user_id == user.uid:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    
    db = get_db()
    
    try:
        # Get existing user
        doc = await db.collection(settings.USERS_COLLECTION).document(user_id).get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail="User not found")
        
        existing_data = doc.to_dict()
        
        # Verify user belongs to same organization
        if existing_data.get("org_id") != user.org_id:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Cannot delete superusers
        if existing_data.get("role") == "superuser":
            raise HTTPException(status_code=403, detail="Cannot delete superuser accounts")
        
        username = existing_data.get("username", user_id)
        
        # Delete from Firebase Auth
        try:
            from ..services.firebase_auth import _init_firebase_admin
            from firebase_admin import auth as firebase_auth
            _init_firebase_admin()
            firebase_auth.delete_user(user_id)
            logger.info(f"Deleted Firebase user: {user_id}")
        except Exception as e:
            logger.warning(f"Failed to delete Firebase user (may not exist): {e}")
        
        # Delete Firestore document
        await db.collection(settings.USERS_COLLECTION).document(user_id).delete()
        logger.info(f"Deleted user document: {user_id}")
        
        # Log activity
        background_tasks.add_task(
            activity_provider.log_activity,
            user.org_id,
            "user_deleted",
            user.uid,
            "user",
            user_id,
            {"username": username}
        )
        
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"User deleted: {user_id} org={user.org_id} | {elapsed:.1f}ms")
        
        return {
            "status": "success",
            "message": "User deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete user")


@router.post("/{user_id}/reset-password")
@limiter.limit(settings.RATE_LIMIT_LOGIN)
async def reset_user_password(
    request: Request,
    user_id: str,
    background_tasks: BackgroundTasks,
    user: UserContext = Depends(require_superuser)
):
    """
    Reset a user's password.
    
    MULTI-TENANCY: Only resets password for users in the superuser's organization.
    Requires superuser role.
    """
    start_time = time.perf_counter()
    
    validate_no_null_bytes(user_id, "user_id")
    if len(user_id) > 100:
        raise HTTPException(status_code=400, detail="Invalid user ID")
    
    req = await request.json()
    new_password = req.get("new_password", "")
    
    validate_no_null_bytes(new_password, "new_password")
    if not new_password or len(new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    
    db = get_db()
    
    try:
        # Get existing user
        doc = await db.collection(settings.USERS_COLLECTION).document(user_id).get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail="User not found")
        
        existing_data = doc.to_dict()
        
        # Verify user belongs to same organization
        if existing_data.get("org_id") != user.org_id:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Cannot reset superuser password (they should use password reset flow)
        if existing_data.get("role") == "superuser":
            raise HTTPException(status_code=403, detail="Cannot reset superuser password through this endpoint")
        
        # Update password in Firebase Auth
        try:
            from ..services.firebase_auth import _init_firebase_admin
            from firebase_admin import auth as firebase_auth
            _init_firebase_admin()
            firebase_auth.update_user(user_id, password=new_password)
            logger.info(f"Reset password for Firebase user: {user_id}")
        except Exception as e:
            logger.error(f"Failed to reset password: {e}")
            raise HTTPException(status_code=500, detail="Failed to reset password")
        
        # Update Firestore document timestamp
        await db.collection(settings.USERS_COLLECTION).document(user_id).update({
            "updated_at": datetime.now(timezone.utc),
            "password_reset_at": datetime.now(timezone.utc),
            "password_reset_by": user.uid
        })
        
        # Log activity
        background_tasks.add_task(
            activity_provider.log_activity,
            user.org_id,
            "password_reset",
            user.uid,
            "user",
            user_id,
            {"by": user.username}
        )
        
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"Password reset: {user_id} by {user.uid} | {elapsed:.1f}ms")
        
        return {
            "status": "success",
            "message": "Password reset successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset password for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset password")
