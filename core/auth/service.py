"""Authentication service with password hashing and token management"""

import hashlib
import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional, Tuple

from .models import APIToken, User, UserRole
from .storage import UserStorage

logger = logging.getLogger(__name__)


class AuthService:
    """Authentication service"""
    
    def __init__(self, storage: UserStorage):
        self.storage = storage
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using SHA-256 with salt"""
        salt = secrets.token_hex(16)
        pwd_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}${pwd_hash}"
    
    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify a password against its hash"""
        try:
            salt, pwd_hash = password_hash.split("$")
            return hashlib.sha256((password + salt).encode()).hexdigest() == pwd_hash
        except Exception:
            return False
    
    @staticmethod
    def generate_token() -> str:
        """Generate a secure random token"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def generate_user_id() -> str:
        """Generate a unique user ID"""
        return f"user_{secrets.token_hex(8)}"
    
    async def register_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.USER,
    ) -> Tuple[Optional[User], Optional[str]]:
        """
        Register a new user.
        
        Returns:
            Tuple of (User, error_message). If successful, error_message is None.
        """
        try:
            # Validate inputs
            if not username or len(username) < 3:
                return None, "Username must be at least 3 characters"
            
            if not email or "@" not in email:
                return None, "Invalid email address"
            
            if not password or len(password) < 6:
                return None, "Password must be at least 6 characters"
            
            # Create user
            user = User(
                user_id=self.generate_user_id(),
                username=username,
                email=email,
                password_hash=self.hash_password(password),
                role=role,
            )
            
            success = await self.storage.create_user(user)
            if not success:
                return None, "Username or email already exists"
            
            logger.info(f"Registered new user: {username}")
            return user, None
            
        except Exception as e:
            logger.error(f"Failed to register user: {e}")
            return None, f"Registration failed: {str(e)}"
    
    async def authenticate_user(
        self, username: str, password: str
    ) -> Tuple[Optional[User], Optional[str]]:
        """
        Authenticate a user by username and password.
        
        Returns:
            Tuple of (User, error_message). If successful, error_message is None.
        """
        try:
            user = await self.storage.get_user_by_username(username)
            
            if not user:
                return None, "Invalid username or password"
            
            if not user.is_active:
                return None, "User account is disabled"
            
            if not self.verify_password(password, user.password_hash):
                return None, "Invalid username or password"
            
            logger.info(f"User authenticated: {username}")
            return user, None
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return None, "Authentication failed"
    
    async def create_api_token(
        self,
        user_id: str,
        token_name: str,
        expires_in_days: Optional[int] = None,
    ) -> Optional[APIToken]:
        """Create a new API token for a user"""
        try:
            token_string = self.generate_token()
            expires_at = None
            
            if expires_in_days:
                expires_at = (
                    datetime.utcnow() + timedelta(days=expires_in_days)
                ).isoformat()
            
            token = APIToken(
                token=token_string,
                user_id=user_id,
                name=token_name,
                expires_at=expires_at,
            )
            
            success = await self.storage.create_token(token)
            if not success:
                return None
            
            logger.info(f"Created API token for user {user_id}: {token_name}")
            return token
            
        except Exception as e:
            logger.error(f"Failed to create API token: {e}")
            return None
    
    async def validate_token(self, token_string: str) -> Tuple[Optional[User], Optional[str]]:
        """
        Validate an API token and return the associated user.
        
        Returns:
            Tuple of (User, error_message). If successful, error_message is None.
        """
        try:
            token = await self.storage.get_token(token_string)
            
            if not token:
                return None, "Invalid token"
            
            if not token.is_active:
                return None, "Token is disabled"
            
            # Check expiration
            if token.expires_at:
                expires_at = datetime.fromisoformat(token.expires_at)
                if datetime.utcnow() > expires_at:
                    return None, "Token has expired"
            
            # Get user
            user = await self.storage.get_user(token.user_id)
            if not user:
                return None, "User not found"
            
            if not user.is_active:
                return None, "User account is disabled"
            
            # Update last used time
            token.last_used_at = datetime.utcnow().isoformat()
            await self.storage.update_token(token)
            
            return user, None
            
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return None, "Token validation failed"
    
    async def revoke_token(self, token_string: str, user_id: str) -> bool:
        """Revoke an API token (user can only revoke their own tokens)"""
        try:
            token = await self.storage.get_token(token_string)
            
            if not token:
                return False
            
            # Verify ownership
            if token.user_id != user_id:
                logger.warning(f"User {user_id} attempted to revoke token owned by {token.user_id}")
                return False
            
            return await self.storage.delete_token(token_string)
            
        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")
            return False
    
    async def list_user_tokens(self, user_id: str) -> list:
        """List all tokens for a user"""
        tokens = await self.storage.get_user_tokens(user_id)
        return [token.to_public_dict() for token in tokens]
    
    async def change_password(
        self, user_id: str, old_password: str, new_password: str
    ) -> Tuple[bool, Optional[str]]:
        """Change user password"""
        try:
            user = await self.storage.get_user(user_id)
            
            if not user:
                return False, "User not found"
            
            if not self.verify_password(old_password, user.password_hash):
                return False, "Invalid current password"
            
            if len(new_password) < 6:
                return False, "New password must be at least 6 characters"
            
            user.password_hash = self.hash_password(new_password)
            user.updated_at = datetime.utcnow().isoformat()
            
            success = await self.storage.update_user(user)
            if not success:
                return False, "Failed to update password"
            
            logger.info(f"Password changed for user {user_id}")
            return True, None
            
        except Exception as e:
            logger.error(f"Failed to change password: {e}")
            return False, "Failed to change password"

