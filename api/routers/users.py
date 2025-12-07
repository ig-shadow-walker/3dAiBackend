"""User management and authentication endpoints"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr

from api.dependencies import get_auth_service, get_current_user
from core.auth.models import User, UserRole
from core.auth.service import AuthService

router = APIRouter()
logger = logging.getLogger(__name__)


# Request/Response Models

class RegisterRequest(BaseModel):
    """User registration request"""
    username: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    """User login request"""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response with API token"""
    user: dict
    token: str
    token_name: str
    message: str

class RegisterResponse(BaseModel):
    """Register response with API token"""
    user: dict
    token: str
    token_name: str
    message: str


class CreateTokenRequest(BaseModel):
    """Create API token request"""
    token_name: str
    expires_in_days: Optional[int] = None


class ChangePasswordRequest(BaseModel):
    """Change password request"""
    old_password: str
    new_password: str


# Endpoints

@router.post("/register", response_model=RegisterResponse, summary="Register a new user")
async def register(
    request: RegisterRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Register a new user account.
    
    Returns a user object without creating an API token.
    Use /login to get an API token after registration.
    """
    try:
        user, error = await auth_service.register_user(
            username=request.username,
            email=request.email,
            password=request.password,
        )
        
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        # Create API token
        token = await auth_service.create_api_token(
            user_id=user.user_id,
            token_name=f"Login token - {request.username}",
            expires_in_days=365,  # 1 year expiration
        )
        
        if not token:
            raise HTTPException(status_code=500, detail="Failed to create API token")

        # return {
        #     "success": True,
        #     "message": "User registered successfully. Please login to get an API token.",
        #     "user": user.to_public_dict(),
        # }
        return RegisterResponse(
            user=user.to_public_dict(),
            token=token.token,
            token_name=token.name,
            message="Login successful. Use this token in Authorization header.",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


@router.post("/login", response_model=LoginResponse, summary="Login and get API token")
async def login(
    request: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Authenticate with username and password, and receive an API token.
    
    The returned token should be used in the Authorization header for subsequent requests:
    `Authorization: Bearer <token>`
    """
    try:
        # Authenticate user
        user, error = await auth_service.authenticate_user(
            username=request.username,
            password=request.password,
        )
        
        if error:
            raise HTTPException(status_code=401, detail=error)
        
        # Create API token
        token = await auth_service.create_api_token(
            user_id=user.user_id,
            token_name=f"Login token - {request.username}",
            expires_in_days=365,  # 1 year expiration
        )
        
        if not token:
            raise HTTPException(status_code=500, detail="Failed to create API token")
        
        return LoginResponse(
            user=user.to_public_dict(),
            token=token.token,
            token_name=token.name,
            message="Login successful. Use this token in Authorization header.",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


@router.get("/me", summary="Get current user profile")
async def get_profile(current_user: User = Depends(get_current_user)):
    """Get the profile of the currently authenticated user"""
    return {
        "success": True,
        "user": current_user.to_public_dict(),
    }


@router.put("/me/password", summary="Change password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
):
    """Change the password for the currently authenticated user"""
    try:
        success, error = await auth_service.change_password(
            user_id=current_user.user_id,
            old_password=request.old_password,
            new_password=request.new_password,
        )
        
        if not success:
            raise HTTPException(status_code=400, detail=error or "Failed to change password")
        
        return {
            "success": True,
            "message": "Password changed successfully",
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(status_code=500, detail="Failed to change password")


@router.post("/tokens", summary="Create a new API token")
async def create_token(
    request: CreateTokenRequest,
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
):
    """Create a new API token for the current user"""
    try:
        token = await auth_service.create_api_token(
            user_id=current_user.user_id,
            token_name=request.token_name,
            expires_in_days=request.expires_in_days,
        )
        
        if not token:
            raise HTTPException(status_code=500, detail="Failed to create token")
        
        return {
            "success": True,
            "message": "API token created successfully",
            "token": token.token,
            "token_info": token.to_public_dict(),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token creation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create token")


@router.get("/tokens", summary="List user's API tokens")
async def list_tokens(
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
):
    """List all API tokens for the current user"""
    try:
        tokens = await auth_service.list_user_tokens(current_user.user_id)
        
        return {
            "success": True,
            "tokens": tokens,
            "total_count": len(tokens),
        }
        
    except Exception as e:
        logger.error(f"Error listing tokens: {e}")
        raise HTTPException(status_code=500, detail="Failed to list tokens")


@router.delete("/tokens/{token_prefix}", summary="Revoke an API token")
async def revoke_token(
    token_prefix: str,
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    Revoke an API token. 
    
    You can provide either the full token or just the first 8 characters (prefix).
    """
    try:
        # Get all user tokens
        all_tokens = await auth_service.storage.get_user_tokens(current_user.user_id)
        
        # Find matching token
        target_token = None
        for token in all_tokens:
            if token.token == token_prefix or token.token.startswith(token_prefix):
                target_token = token.token
                break
        
        if not target_token:
            raise HTTPException(status_code=404, detail="Token not found")
        
        success = await auth_service.revoke_token(target_token, current_user.user_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to revoke token")
        
        return {
            "success": True,
            "message": "Token revoked successfully",
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token revocation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to revoke token")


# Admin endpoints

@router.get("/admin/users", summary="List all users (Admin only)")
async def list_users(
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
):
    """List all users (admin only)"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        users = await auth_service.storage.list_users(limit=limit, offset=offset)
        
        return {
            "success": True,
            "users": [user.to_public_dict() for user in users],
            "count": len(users),
            "pagination": {
                "limit": limit,
                "offset": offset,
            },
        }
        
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(status_code=500, detail="Failed to list users")


@router.get("/admin/users/{user_id}", summary="Get user by ID (Admin only)")
async def get_user(
    user_id: str,
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
):
    """Get user details by ID (admin only)"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        user = await auth_service.storage.get_user(user_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "success": True,
            "user": user.to_public_dict(),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user")


@router.delete("/admin/users/{user_id}", summary="Delete user (Admin only)")
async def delete_user(
    user_id: str,
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
):
    """Delete a user (admin only)"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    if user_id == current_user.user_id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    
    try:
        success = await auth_service.storage.delete_user(user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "success": True,
            "message": f"User {user_id} deleted successfully",
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete user")

