"""
Authentication router for admin backend.

MULTI-TENANCY: Supports both legacy single-admin auth and Firebase multi-user auth.
"""
import time
from datetime import datetime, timezone
from fastapi import APIRouter, Request, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from ..config import settings, logger
from ..utils.validators import validate_no_null_bytes
from ..services.auth import verify_credentials, create_jwt_token
from ..dependencies import get_current_user, UserContext
from ..utils.limiter import limiter

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class LoginRequest(BaseModel):
    """Login request model - supports both username and email"""
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: str


class FirebaseLoginRequest(BaseModel):
    """Firebase login request model"""
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    """Registration request model for new organizations"""
    email: EmailStr
    password: str = Field(..., min_length=8)
    organisation_id: str = Field(..., min_length=3, max_length=50)
    organisation_name: str = Field(..., min_length=1, max_length=200)
    username: Optional[str] = None  # Optional, defaults to org_id


@router.post("/login")
@limiter.limit(settings.RATE_LIMIT_LOGIN)
async def login(request: Request):
    """
    Unified login endpoint supporting both legacy admin and Firebase users.
    
    Accepts:
    - username/password: Tries legacy admin first, then Firebase org_id login
    - email/password: Routes to Firebase authentication
    
    SECURITY: Returns same error for both invalid username and password
    to prevent user enumeration attacks.
    """
    start_time = time.perf_counter()
    req = await request.json()
    username = req.get("username", "")
    email = req.get("email", "")
    password = req.get("password", "")
    
    # Use email if provided, otherwise use username
    identifier = email or username
    
    validate_no_null_bytes(identifier, "identifier")
    validate_no_null_bytes(password, "password")
    
    if not identifier or len(identifier) > 254 or not password or len(password) > 200:
        raise HTTPException(status_code=400, detail="Invalid input")
    
    # If it looks like an email, use Firebase login directly
    if "@" in identifier:
        try:
            from ..services.firebase_auth import firebase_auth_service
            result = await firebase_auth_service.authenticate_user(identifier, password)
            
            if result["success"]:
                user_context = result["user"]
                token = create_jwt_token(
                    username=user_context.username,
                    role=user_context.role,
                    org_id=user_context.org_id,
                    email=user_context.email,
                    uid=user_context.uid
                )
                elapsed = (time.perf_counter() - start_time) * 1000
                logger.info(f"Firebase login successful (email): {user_context.uid} org={user_context.org_id} | {elapsed:.1f}ms")
                return {
                    "status": "success",
                    "message": "Login successful",
                    "user": {
                        "uid": user_context.uid,
                        "username": user_context.username,
                        "email": user_context.email,
                        "org_id": user_context.org_id,
                        "role": user_context.role,
                        "token": token
                    }
                }
            else:
                raise HTTPException(status_code=401, detail="Invalid credentials")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Firebase login error: {e}")
            raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # For non-email identifiers (org_id/username):
    # 1. Try legacy admin first
    valid, err = verify_credentials(identifier, password)
    if valid:
        token = create_jwt_token(identifier, role="admin", org_id="default", uid=identifier)
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"Legacy login successful: {identifier} | {elapsed:.1f}ms")
        return {
            "status": "success", 
            "message": "Login successful", 
            "user": {
                "username": identifier, 
                "role": "admin", 
                "org_id": "default",
                "token": token
            }
        }
    
    # 2. Try Firebase org_id login - lookup user by org_id (uid)
    try:
        from ..services.firebase_auth import firebase_auth_service
        from ..providers.database.firestore_init import get_db
        
        db = get_db()
        # Look up user document by uid (which is the org_id for superusers)
        user_doc = await db.collection(settings.USERS_COLLECTION).document(identifier).get()
        
        if user_doc.exists:
            user_data = user_doc.to_dict()
            user_email = user_data.get("email")
            
            if user_email:
                # Authenticate using email from user record
                result = await firebase_auth_service.authenticate_user(user_email, password)
                
                if result["success"]:
                    user_context = result["user"]
                    token = create_jwt_token(
                        username=user_context.username,
                        role=user_context.role,
                        org_id=user_context.org_id,
                        email=user_context.email,
                        uid=user_context.uid
                    )
                    elapsed = (time.perf_counter() - start_time) * 1000
                    logger.info(f"Firebase login successful (org_id): {user_context.uid} org={user_context.org_id} | {elapsed:.1f}ms")
                    return {
                        "status": "success",
                        "message": "Login successful",
                        "user": {
                            "uid": user_context.uid,
                            "username": user_context.username,
                            "email": user_context.email,
                            "org_id": user_context.org_id,
                            "role": user_context.role,
                            "token": token
                        }
                    }
    except Exception as e:
        logger.warning(f"Firebase org_id login fallback failed: {e}")
    
    # All auth methods failed
    raise HTTPException(status_code=401, detail="Invalid credentials")


@router.post("/firebase-login")
@limiter.limit(settings.RATE_LIMIT_LOGIN)
async def firebase_login(request: Request):
    """
    Firebase authentication endpoint for multi-tenant access.
    
    Verifies credentials against Firebase Auth and returns JWT with org_id.
    Supports organizations with format: {org_id}_{username}
    """
    start_time = time.perf_counter()
    
    try:
        from ..services.firebase_auth import firebase_auth_service
    except ImportError:
        raise HTTPException(500, "Firebase auth service not configured")
    
    req = await request.json()
    email = req.get("email", "")
    password = req.get("password", "")
    
    validate_no_null_bytes(email, "email")
    validate_no_null_bytes(password, "password")
    
    if not email or len(email) > 254:
        raise HTTPException(status_code=400, detail="Invalid email")
    if not password or len(password) > 200:
        raise HTTPException(status_code=400, detail="Invalid password")
    
    # Verify with Firebase
    result = await firebase_auth_service.authenticate_user(email, password)
    
    if not result["success"]:
        logger.warning(f"Firebase login failed for {email}: {result.get('error')}")
        raise HTTPException(status_code=401, detail=result.get("error", "Invalid credentials"))
    
    user_context = result["user"]
    
    # Create JWT token with org_id and uid
    token = create_jwt_token(
        username=user_context.username,
        role=user_context.role,
        org_id=user_context.org_id,
        email=user_context.email,
        uid=user_context.uid
    )
    
    elapsed = (time.perf_counter() - start_time) * 1000
    logger.info(f"Firebase login successful: {user_context.uid} org={user_context.org_id} | {elapsed:.1f}ms")
    
    return {
        "status": "success",
        "message": "Login successful",
        "user": {
            "uid": user_context.uid,
            "username": user_context.username,
            "email": user_context.email,
            "org_id": user_context.org_id,
            "role": user_context.role,
            "token": token
        }
    }


@router.post("/refresh")
async def refresh_token(user: UserContext = Depends(get_current_user)):
    """
    Refresh JWT token.
    
    MULTI-TENANCY: Preserves org_id in refreshed token.
    """
    start_time = time.perf_counter()
    new_token = create_jwt_token(
        username=user.username, 
        role=user.role,
        org_id=user.org_id,
        email=user.email,
        uid=user.uid
    )
    elapsed = (time.perf_counter() - start_time) * 1000
    logger.info(f"Token refresh: {user.username} org={user.org_id} | {elapsed:.1f}ms")
    return {"status": "success", "token": new_token}


@router.get("/me")
async def get_current_user_info(user: UserContext = Depends(get_current_user)):
    """
    Get current user information.
    
    Returns user's profile including organization and role.
    """
    return {
        "status": "success",
        "user": {
            "uid": user.uid,
            "username": user.username,
            "email": user.email,
            "org_id": user.org_id,
            "role": user.role,
            "can_write": user.can_write,
            "can_read": user.can_read,
            "is_superuser": user.is_superuser
        }
    }


@router.post("/logout")
async def logout():
    """
    Logout endpoint.
    
    Note: With JWT, logout is client-side (token deletion).
    Server-side logout would require token blacklisting.
    """
    return {"status": "success", "message": "Logged out successfully"}


@router.get("/check-org-id")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def check_org_id_availability(request: Request, org_id: str):
    """
    Check if an organization ID is available.
    
    Returns:
        {"available": true/false, "org_id": "..."}
    """
    from ..providers.database.firestore_init import get_db
    
    if not org_id or len(org_id) < 3:
        raise HTTPException(status_code=400, detail="Organisation ID must be at least 3 characters")
    
    # Validate org_id format (alphanumeric + underscores/hyphens only)
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', org_id):
        raise HTTPException(
            status_code=400, 
            detail="Organisation ID can only contain letters, numbers, hyphens, and underscores"
        )
    
    validate_no_null_bytes(org_id, "org_id")
    
    db = get_db()
    
    try:
        # Check if organization exists
        org_doc = await db.collection(settings.ORGANIZATIONS_COLLECTION).document(org_id).get()
        available = not org_doc.exists
        
        logger.info(f"Org ID availability check: {org_id} - {'available' if available else 'taken'}")
        
        return {
            "available": available,
            "org_id": org_id
        }
    except Exception as e:
        logger.error(f"Error checking org ID availability: {e}")
        raise HTTPException(status_code=500, detail="Failed to check availability")


@router.post("/register")
@limiter.limit(settings.RATE_LIMIT_LOGIN)
async def register(request: Request):
    """
    Register a new organization and superuser.
    
    Creates:
    1. Firebase Auth user with email/password
    2. Organization document in Firestore
    3. User document in Firestore
    4. Returns JWT token for immediate login
    
    The first user in an organization is automatically a superuser.
    """
    start_time = time.perf_counter()
    
    try:
        from ..services.firebase_auth import firebase_auth_service, _init_firebase_admin
        from firebase_admin import auth as firebase_auth
    except ImportError:
        raise HTTPException(500, "Firebase auth service not configured")
    
    req = await request.json()
    email = req.get("email", "").strip()
    password = req.get("password", "")
    org_id = req.get("organisation_id", "").strip().lower()
    org_name = req.get("organisation_name", "").strip()
    username = req.get("username", org_id).strip()  # Default to org_id
    
    # Validate inputs
    validate_no_null_bytes(email, "email")
    validate_no_null_bytes(password, "password")
    validate_no_null_bytes(org_id, "organisation_id")
    validate_no_null_bytes(org_name, "organisation_name")
    
    if not email or len(email) > 254:
        raise HTTPException(status_code=400, detail="Invalid email address")
    if not password or len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if not org_id or len(org_id) < 3 or len(org_id) > 50:
        raise HTTPException(status_code=400, detail="Organisation ID must be 3-50 characters")
    if not org_name or len(org_name) > 200:
        raise HTTPException(status_code=400, detail="Organisation name is required")
    
    # Validate org_id format
    import re
    if not re.match(r'^[a-z0-9_-]+$', org_id):
        raise HTTPException(
            status_code=400,
            detail="Organisation ID can only contain lowercase letters, numbers, hyphens, and underscores"
        )
    
    from ..providers.database.firestore_init import get_db
    db = get_db()
    
    try:
        # Check if organization already exists
        org_doc = await db.collection(settings.ORGANIZATIONS_COLLECTION).document(org_id).get()
        if org_doc.exists:
            raise HTTPException(status_code=409, detail="Organisation ID already taken")
        
        # Initialize Firebase Admin
        _init_firebase_admin()
        
        # Create Firebase Auth user
        # UID format: {org_id} for superuser (no underscore)
        uid = org_id
        
        try:
            firebase_user = firebase_auth.create_user(
                uid=uid,
                email=email,
                password=password,
                display_name=org_name,
                email_verified=False
            )
            logger.info(f"Created Firebase user: {uid} ({email})")
        except Exception as e:
            error_msg = str(e)
            if "EMAIL_EXISTS" in error_msg or "already exists" in error_msg.lower():
                raise HTTPException(status_code=409, detail="Email already registered")
            elif "UID_ALREADY_EXISTS" in error_msg:
                raise HTTPException(status_code=409, detail="Organisation ID already taken")
            else:
                logger.error(f"Firebase user creation failed: {e}")
                raise HTTPException(status_code=500, detail=f"User creation failed: {error_msg}")
        
        # Create organization document
        now = datetime.now(timezone.utc)
        org_data = {
            "id": org_id,
            "name": org_name,
            "created_at": now,
            "status": "active",
            "superuser_email": email,
            "created_by": uid
        }
        await db.collection(settings.ORGANIZATIONS_COLLECTION).document(org_id).set(org_data)
        logger.info(f"Created organization: {org_id}")
        
        # Create user document
        user_data = {
            "uid": uid,
            "username": username,
            "email": email,
            "org_id": org_id,
            "role": "superuser",
            "created_at": now,
            "status": "active"
        }
        await db.collection(settings.USERS_COLLECTION).document(uid).set(user_data)
        logger.info(f"Created user document: {uid}")
        
        # Create JWT token
        token = create_jwt_token(
            username=username,
            role="superuser",
            org_id=org_id,
            email=email,
            uid=uid
        )
        
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"Registration successful: {uid} org={org_id} | {elapsed:.1f}ms")
        
        return {
            "status": "success",
            "message": "Registration successful",
            "user": {
                "uid": uid,
                "username": username,
                "email": email,
                "org_id": org_id,
                "role": "superuser",
                "token": token
            }
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")
