"""User and authentication models"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class UserRole(str, Enum):
    """User role enumeration"""
    USER = "user"
    ADMIN = "admin"


@dataclass
class User:
    """User model"""
    user_id: str
    username: str
    email: str
    password_hash: str
    role: UserRole = UserRole.USER
    is_active: bool = True
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "password_hash": self.password_hash,
            "role": self.role.value if isinstance(self.role, UserRole) else self.role,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }
    
    def to_public_dict(self) -> dict:
        """Convert to public dictionary (without sensitive data)"""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value if isinstance(self.role, UserRole) else self.role,
            "is_active": self.is_active,
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "User":
        """Create user from dictionary"""
        role = data.get("role", UserRole.USER)
        if isinstance(role, str):
            role = UserRole(role)
        
        return cls(
            user_id=data["user_id"],
            username=data["username"],
            email=data["email"],
            password_hash=data["password_hash"],
            role=role,
            is_active=data.get("is_active", True),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class APIToken:
    """API token model"""
    token: str
    user_id: str
    name: str  # Human-readable name for the token
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: Optional[str] = None
    last_used_at: Optional[str] = None
    is_active: bool = True
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "token": self.token,
            "user_id": self.user_id,
            "name": self.name,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_used_at": self.last_used_at,
            "is_active": self.is_active,
        }
    
    def to_public_dict(self) -> dict:
        """Convert to public dictionary (without full token)"""
        return {
            "token_prefix": self.token[:8] + "..." if len(self.token) > 8 else "***",
            "name": self.name,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_used_at": self.last_used_at,
            "is_active": self.is_active,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "APIToken":
        """Create token from dictionary"""
        return cls(
            token=data["token"],
            user_id=data["user_id"],
            name=data["name"],
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            expires_at=data.get("expires_at"),
            last_used_at=data.get("last_used_at"),
            is_active=data.get("is_active", True),
        )

