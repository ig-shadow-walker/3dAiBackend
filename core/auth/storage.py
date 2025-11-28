"""User storage layer using Redis"""

import json
import logging
from typing import List, Optional

from redis.asyncio import Redis

from .models import APIToken, User

logger = logging.getLogger(__name__)


class UserStorage:
    """
    Redis-based user storage.
    
    Redis Key Schema:
        - user:{user_id} -> User JSON
        - user:username:{username} -> user_id
        - user:email:{email} -> user_id
        - token:{token} -> APIToken JSON
        - user:{user_id}:tokens -> Set of token strings
        - users:all -> Set of all user_ids
    """
    
    def __init__(self, redis_client: Redis, key_prefix: str = "3daigc"):
        self.redis = redis_client
        self.prefix = key_prefix
    
    def _key(self, *parts) -> str:
        """Generate Redis key with prefix"""
        return ":".join([self.prefix] + list(parts))
    
    # User operations
    
    async def create_user(self, user: User) -> bool:
        """Create a new user"""
        try:
            # Check if username or email already exists
            existing_by_username = await self.get_user_by_username(user.username)
            if existing_by_username:
                logger.warning(f"Username {user.username} already exists")
                return False
            
            existing_by_email = await self.get_user_by_email(user.email)
            if existing_by_email:
                logger.warning(f"Email {user.email} already exists")
                return False
            
            # Store user data
            user_key = self._key("user", user.user_id)
            username_key = self._key("user", "username", user.username)
            email_key = self._key("user", "email", user.email)
            all_users_key = self._key("users", "all")
            
            # Use pipeline for atomic operations
            async with self.redis.pipeline() as pipe:
                pipe.set(user_key, json.dumps(user.to_dict()))
                pipe.set(username_key, user.user_id)
                pipe.set(email_key, user.user_id)
                pipe.sadd(all_users_key, user.user_id)
                await pipe.execute()
            
            logger.info(f"Created user: {user.username} ({user.user_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            return False
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        try:
            user_key = self._key("user", user_id)
            user_data = await self.redis.get(user_key)
            
            if not user_data:
                return None
            
            return User.from_dict(json.loads(user_data))
            
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {e}")
            return None
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        try:
            username_key = self._key("user", "username", username)
            user_id = await self.redis.get(username_key)
            
            if not user_id:
                return None
            
            return await self.get_user(user_id.decode() if isinstance(user_id, bytes) else user_id)
            
        except Exception as e:
            logger.error(f"Failed to get user by username {username}: {e}")
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        try:
            email_key = self._key("user", "email", email)
            user_id = await self.redis.get(email_key)
            
            if not user_id:
                return None
            
            return await self.get_user(user_id.decode() if isinstance(user_id, bytes) else user_id)
            
        except Exception as e:
            logger.error(f"Failed to get user by email {email}: {e}")
            return None
    
    async def update_user(self, user: User) -> bool:
        """Update user data"""
        try:
            user_key = self._key("user", user.user_id)
            
            # Check if user exists
            if not await self.redis.exists(user_key):
                logger.warning(f"User {user.user_id} does not exist")
                return False
            
            await self.redis.set(user_key, json.dumps(user.to_dict()))
            logger.info(f"Updated user: {user.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update user {user.user_id}: {e}")
            return False
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete a user"""
        try:
            user = await self.get_user(user_id)
            if not user:
                return False
            
            user_key = self._key("user", user_id)
            username_key = self._key("user", "username", user.username)
            email_key = self._key("user", "email", user.email)
            all_users_key = self._key("users", "all")
            
            # Delete all user tokens
            tokens = await self.get_user_tokens(user_id)
            for token in tokens:
                await self.delete_token(token.token)
            
            # Delete user data
            async with self.redis.pipeline() as pipe:
                pipe.delete(user_key)
                pipe.delete(username_key)
                pipe.delete(email_key)
                pipe.srem(all_users_key, user_id)
                await pipe.execute()
            
            logger.info(f"Deleted user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete user {user_id}: {e}")
            return False
    
    async def list_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """List all users"""
        try:
            all_users_key = self._key("users", "all")
            user_ids = await self.redis.smembers(all_users_key)
            
            users = []
            for user_id in user_ids:
                user_id_str = user_id.decode() if isinstance(user_id, bytes) else user_id
                user = await self.get_user(user_id_str)
                if user:
                    users.append(user)
            
            # Sort by created_at and apply pagination
            users.sort(key=lambda u: u.created_at, reverse=True)
            return users[offset:offset + limit]
            
        except Exception as e:
            logger.error(f"Failed to list users: {e}")
            return []
    
    # Token operations
    
    async def create_token(self, token: APIToken) -> bool:
        """Create a new API token"""
        try:
            token_key = self._key("token", token.token)
            user_tokens_key = self._key("user", token.user_id, "tokens")
            
            # Store token data
            async with self.redis.pipeline() as pipe:
                pipe.set(token_key, json.dumps(token.to_dict()))
                pipe.sadd(user_tokens_key, token.token)
                await pipe.execute()
            
            logger.info(f"Created token for user {token.user_id}: {token.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create token: {e}")
            return False
    
    async def get_token(self, token: str) -> Optional[APIToken]:
        """Get token by token string"""
        try:
            token_key = self._key("token", token)
            token_data = await self.redis.get(token_key)
            
            if not token_data:
                return None
            
            return APIToken.from_dict(json.loads(token_data))
            
        except Exception as e:
            logger.error(f"Failed to get token: {e}")
            return None
    
    async def update_token(self, token: APIToken) -> bool:
        """Update token data (e.g., last_used_at)"""
        try:
            token_key = self._key("token", token.token)
            
            if not await self.redis.exists(token_key):
                return False
            
            await self.redis.set(token_key, json.dumps(token.to_dict()))
            return True
            
        except Exception as e:
            logger.error(f"Failed to update token: {e}")
            return False
    
    async def delete_token(self, token: str) -> bool:
        """Delete a token"""
        try:
            token_obj = await self.get_token(token)
            if not token_obj:
                return False
            
            token_key = self._key("token", token)
            user_tokens_key = self._key("user", token_obj.user_id, "tokens")
            
            async with self.redis.pipeline() as pipe:
                pipe.delete(token_key)
                pipe.srem(user_tokens_key, token)
                await pipe.execute()
            
            logger.info(f"Deleted token: {token[:8]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete token: {e}")
            return False
    
    async def get_user_tokens(self, user_id: str) -> List[APIToken]:
        """Get all tokens for a user"""
        try:
            user_tokens_key = self._key("user", user_id, "tokens")
            token_strings = await self.redis.smembers(user_tokens_key)
            
            tokens = []
            for token_str in token_strings:
                token_str = token_str.decode() if isinstance(token_str, bytes) else token_str
                token = await self.get_token(token_str)
                if token:
                    tokens.append(token)
            
            return tokens
            
        except Exception as e:
            logger.error(f"Failed to get user tokens for {user_id}: {e}")
            return []

