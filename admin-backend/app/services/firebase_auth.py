"""
Firebase Authentication Service for Multi-Tenant Organization System.

This module provides secure authentication using Firebase Auth with:
- Firebase Custom Tokens for session management
- Organization-level data isolation (org_id)
- Role-based access control (superuser, admin, viewer, analyser)
- Password verification via Google Identity REST API

Security Features:
- Passwords verified via Firebase Auth (not stored locally)
- org_id extracted from authenticated user (never from request body)
- All operations scoped to user's organization
"""
import os
import secrets
from datetime import datetime, timezone, timedelta
from typing import Tuple, Optional, Dict, Any
import requests
from pydantic import BaseModel, Field, EmailStr

from ..config import settings, logger

# Try to import firebase-admin (optional - for custom token generation)
try:
    import firebase_admin
    from firebase_admin import auth as firebase_auth, credentials
    FIREBASE_ADMIN_AVAILABLE = True
except ImportError:
    FIREBASE_ADMIN_AVAILABLE = False
    logger.warning("firebase-admin not installed - using REST API only")

# Initialize Firebase Admin SDK if available
_firebase_app = None


def _init_firebase_admin():
    """Initialize Firebase Admin SDK for custom token generation."""
    global _firebase_app
    if not FIREBASE_ADMIN_AVAILABLE or _firebase_app is not None:
        return _firebase_app
    
    try:
        if firebase_admin._apps:
            _firebase_app = firebase_admin.get_app()
            return _firebase_app
        
        # Try credential file first
        if settings.FIREBASE_CRED_PATH and os.path.exists(settings.FIREBASE_CRED_PATH):
            cred = credentials.Certificate(settings.FIREBASE_CRED_PATH)
            _firebase_app = firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin initialized from credential file")
            return _firebase_app
        
        # Try base64 credentials
        if settings.FIREBASE_CRED_BASE64:
            import base64
            import json
            cred_json = base64.b64decode(settings.FIREBASE_CRED_BASE64).decode('utf-8')
            cred_dict = json.loads(cred_json)
            cred = credentials.Certificate(cred_dict)
            _firebase_app = firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin initialized from base64 credentials")
            return _firebase_app
        
        # Try default credentials
        _firebase_app = firebase_admin.initialize_app()
        logger.info("Firebase Admin initialized with default credentials")
        return _firebase_app
        
    except Exception as e:
        logger.error(f"Failed to initialize Firebase Admin: {e}")
        return None


# --- Pydantic Models for Authentication ---

class UserContext(BaseModel):
    """User context extracted from authentication."""
    uid: str
    org_id: str
    role: str
    email: Optional[str] = None
    name: Optional[str] = None
    
    @property
    def is_superuser(self) -> bool:
        return self.role == "superuser"
    
    @property
    def is_admin(self) -> bool:
        return self.role in ["superuser", "admin"]
    
    @property
    def can_write(self) -> bool:
        """Check if user can perform write operations."""
        return self.role in ["superuser", "admin"]
    
    @property
    def can_read(self) -> bool:
        """Check if user can perform read operations."""
        return self.role in ["superuser", "admin", "viewer"]


class LoginRequest(BaseModel):
    """Login request model."""
    user_id: str = Field(..., min_length=1)
    password: str = Field(..., min_length=6)


class LoginResponse(BaseModel):
    """Login response with user context and token."""
    status: str
    message: str
    user: UserContext
    token: str
    expires_at: datetime


# --- Authentication Functions ---

def verify_password_firebase(email: str, password: str) -> Tuple[bool, Optional[dict], str]:
    """
    Verify password using Firebase Identity REST API.
    
    This is the SECURE way to verify passwords - Firebase handles the verification.
    
    Args:
        email: User's email address
        password: Plain text password to verify
    
    Returns:
        Tuple of (success, firebase_response_data, error_message)
    """
    api_key = os.getenv("FIREBASE_WEB_API_KEY")
    if not api_key:
        logger.error("FIREBASE_WEB_API_KEY not configured")
        return False, None, "Server configuration error"
    
    verify_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
    
    try:
        response = requests.post(verify_url, json={
            "email": email,
            "password": password,
            "returnSecureToken": True
        }, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return True, data, ""
        
        # Parse Firebase error
        error_data = response.json()
        error_message = error_data.get("error", {}).get("message", "Authentication failed")
        
        if "INVALID_PASSWORD" in error_message:
            return False, None, "Invalid password"
        elif "EMAIL_NOT_FOUND" in error_message:
            return False, None, "User not found"
        elif "USER_DISABLED" in error_message:
            return False, None, "Account disabled"
        elif "TOO_MANY_ATTEMPTS" in error_message:
            return False, None, "Too many failed attempts. Try again later."
        else:
            return False, None, error_message
            
    except requests.exceptions.Timeout:
        logger.error("Firebase authentication timeout")
        return False, None, "Authentication service timeout"
    except requests.exceptions.RequestException as e:
        logger.error(f"Firebase authentication error: {e}")
        return False, None, "Authentication service unavailable"


def create_custom_token(uid: str, claims: Optional[dict] = None) -> Optional[str]:
    """
    Create Firebase custom token for authenticated session.
    
    Args:
        uid: User ID to encode in token
        claims: Optional custom claims (org_id, role, etc.)
    
    Returns:
        Custom token string or None if failed
    """
    _init_firebase_admin()
    
    if not FIREBASE_ADMIN_AVAILABLE or _firebase_app is None:
        logger.warning("Firebase Admin not available for custom token generation")
        return None
    
    try:
        token = firebase_auth.create_custom_token(uid, claims)
        return token.decode('utf-8') if isinstance(token, bytes) else token
    except Exception as e:
        logger.error(f"Failed to create custom token: {e}")
        return None


def verify_id_token(id_token: str) -> Tuple[bool, Optional[dict], str]:
    """
    Verify Firebase ID token.
    
    Args:
        id_token: Firebase ID token from client
    
    Returns:
        Tuple of (success, decoded_token, error_message)
    """
    _init_firebase_admin()
    
    if not FIREBASE_ADMIN_AVAILABLE or _firebase_app is None:
        return False, None, "Firebase Admin not available"
    
    try:
        decoded = firebase_auth.verify_id_token(id_token)
        return True, decoded, ""
    except firebase_auth.ExpiredIdTokenError:
        return False, None, "Token expired"
    except firebase_auth.RevokedIdTokenError:
        return False, None, "Token revoked"
    except firebase_auth.InvalidIdTokenError as e:
        return False, None, f"Invalid token: {e}"
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return False, None, "Token verification failed"


def extract_org_id_from_uid(uid: str) -> str:
    """
    Extract organization ID from user ID.
    
    User ID format:
    - Superuser: "org_id" (no underscore in org part)
    - Sub-user: "org_id_username"
    
    Args:
        uid: User ID (e.g., "aot169" or "aot169_john")
    
    Returns:
        Organization ID
    """
    if "_" in uid:
        # Sub-user: first part before underscore is org_id
        return uid.split("_")[0]
    else:
        # Superuser: uid IS the org_id
        return uid


def get_user_from_firestore(db, user_id: str) -> Optional[dict]:
    """
    Get user data from Firestore users collection.
    
    This is a synchronous helper for getting user data during auth.
    
    Args:
        db: Firestore client (sync)
        user_id: User ID to look up
    
    Returns:
        User data dict or None
    """
    try:
        doc = db.collection("users").document(user_id).get()
        if doc.exists:
            data = doc.to_dict()
            data['id'] = doc.id
            return data
        return None
    except Exception as e:
        logger.error(f"Failed to get user from Firestore: {e}")
        return None


async def get_user_from_firestore_async(db, user_id: str) -> Optional[dict]:
    """
    Get user data from Firestore users collection (async version).
    
    Args:
        db: Firestore async client
        user_id: User ID to look up
    
    Returns:
        User data dict or None
    """
    try:
        doc = await db.collection("users").document(user_id).get()
        if doc.exists:
            data = doc.to_dict()
            data['id'] = doc.id
            return data
        return None
    except Exception as e:
        logger.error(f"Failed to get user from Firestore: {e}")
        return None


# --- Legacy JWT Support (for backwards compatibility during migration) ---

import jwt
from fastapi.security import HTTPBearer

security = HTTPBearer(auto_error=False)


def create_jwt_token(username: str, role: str = "admin", org_id: str = None) -> str:
    """
    Create JWT token with organization context.
    
    This is the ENHANCED version that includes org_id for multi-tenancy.
    """
    if not username or not isinstance(username, str):
        raise ValueError("Invalid username")
    if len(username) > 100:
        raise ValueError("Username too long")
    
    # Auto-extract org_id if not provided
    if org_id is None:
        org_id = extract_org_id_from_uid(username)
    
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=settings.JWT_EXPIRATION_MINUTES)
    
    payload = {
        "sub": username,
        "role": role,
        "org_id": org_id,  # CRITICAL: Include org_id for multi-tenancy
        "iat": now,
        "exp": exp,
        "jti": secrets.token_hex(16)
    }
    
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_jwt_token(token: str) -> Tuple[Optional[dict], bool, int]:
    """
    Decode JWT token and extract user context.
    
    Returns:
        Tuple of (payload, needs_refresh, remaining_seconds)
    """
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        exp = payload.get("exp", 0)
        now_ts = datetime.now(timezone.utc).timestamp()
        remaining = max(0, int(exp - now_ts))
        needs_refresh = remaining < (settings.JWT_REFRESH_THRESHOLD_MINUTES * 60)
        
        # Ensure org_id is present (backwards compatibility)
        if "org_id" not in payload:
            payload["org_id"] = extract_org_id_from_uid(payload.get("sub", ""))
        
        return payload, needs_refresh, remaining
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None, False, 0


def verify_credentials(username: str, password: str) -> Tuple[bool, str]:
    """
    LEGACY: Verify admin credentials using local config.
    
    This is kept for backwards compatibility. New deployments should use
    Firebase Auth via verify_password_firebase().
    """
    username_match = secrets.compare_digest(username, settings.ADMIN_USERNAME)
    password_match = settings.verify_password(password)
    if not username_match or not password_match:
        return False, "invalid_credentials"
    return True, ""


# --- Firebase Auth Service Class ---

class FirebaseAuthService:
    """
    Firebase Authentication Service for multi-tenant organization system.
    
    Provides user authentication via Firebase with organization-level isolation.
    """
    
    def __init__(self):
        self._db = None
    
    def _get_firestore_client(self):
        """Get Firestore client for user lookups."""
        if self._db is None:
            _init_firebase_admin()
            if FIREBASE_ADMIN_AVAILABLE:
                try:
                    from firebase_admin import firestore
                    self._db = firestore.client()
                except Exception as e:
                    logger.error(f"Failed to get Firestore client: {e}")
        return self._db
    
    async def authenticate_user(self, email: str, password: str) -> Dict[str, Any]:
        """
        Authenticate user with Firebase and return user context.
        
        Args:
            email: User's email address
            password: Plain text password
        
        Returns:
            Dict with success status, user context, and any errors
        """
        # Verify password with Firebase
        success, firebase_data, error_msg = verify_password_firebase(email, password)
        
        if not success:
            return {"success": False, "error": error_msg}
        
        # Get local UID from Firebase response
        local_uid = firebase_data.get("localId", "")
        
        # Try to get user data from Firestore
        db = self._get_firestore_client()
        user_data = None
        
        if db:
            user_data = get_user_from_firestore(db, local_uid)
        
        # Extract org_id and role
        if user_data:
            org_id = user_data.get("org_id", extract_org_id_from_uid(local_uid))
            role = user_data.get("role", "viewer")
            username = user_data.get("username", local_uid)
        else:
            # Fallback: extract from UID
            org_id = extract_org_id_from_uid(local_uid)
            role = "viewer"  # Safe default
            username = local_uid
        
        user_context = UserContext(
            uid=local_uid,
            org_id=org_id,
            role=role,
            email=email,
            name=username
        )
        
        return {
            "success": True,
            "user": type('UserContext', (), {
                'uid': local_uid,
                'username': username,
                'email': email,
                'org_id': org_id,
                'role': role,
                'can_write': role in ("superuser", "admin"),
                'can_read': role in ("superuser", "admin", "viewer"),
                'is_superuser': role == "superuser"
            })()
        }


# Singleton instance
firebase_auth_service = FirebaseAuthService()
