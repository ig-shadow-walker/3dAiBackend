#!/usr/bin/env python3
"""
Script to create an admin user for the 3DAIGC API.

This script should be run once when you first enable user authentication.
It will create an admin user that can manage other users and access all jobs.

Usage:
    python scripts/create_admin_user.py --username admin --email admin@example.com --password your_secure_password
    
Or run interactively:
    python scripts/create_admin_user.py
"""

import argparse
import asyncio
import getpass
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from redis.asyncio import Redis

from core.auth import AuthService, UserStorage
from core.auth.models import UserRole
from core.config import get_settings


async def create_admin_user(username: str, email: str, password: str, redis_url: str):
    """Create an admin user"""
    print(f"\n🔐 Creating admin user: {username}")
    print(f"   Email: {email}")
    print(f"   Connecting to Redis: {redis_url}")
    
    try:
        # Connect to Redis
        redis_client = Redis.from_url(redis_url, decode_responses=True)
        user_storage = UserStorage(redis_client, key_prefix="3daigc")
        auth_service = AuthService(user_storage)
        
        # Create admin user
        user, error = await auth_service.register_user(
            username=username,
            email=email,
            password=password,
            role=UserRole.ADMIN,
        )
        
        if error:
            print(f"\n❌ Error: {error}")
            return False
        
        print(f"\n✅ Admin user created successfully!")
        print(f"   User ID: {user.user_id}")
        print(f"   Username: {user.username}")
        print(f"   Email: {user.email}")
        print(f"   Role: {user.role.value}")
        
        # Create a default API token
        token = await auth_service.create_api_token(
            user_id=user.user_id,
            token_name="Default admin token",
            expires_in_days=365,
        )
        
        if token:
            print(f"\n🔑 API Token created (valid for 1 year):")
            print(f"   {token.token}")
            print(f"\n   Use this token in your requests:")
            print(f"   curl -H 'Authorization: Bearer {token.token}' ...")
        
        # Cleanup
        await redis_client.close()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to create admin user: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create an admin user for the 3DAIGC API"
    )
    parser.add_argument(
        "--username",
        type=str,
        help="Admin username (default: admin)",
    )
    parser.add_argument(
        "--email",
        type=str,
        help="Admin email address",
    )
    parser.add_argument(
        "--password",
        type=str,
        help="Admin password (will prompt if not provided)",
    )
    parser.add_argument(
        "--redis-url",
        type=str,
        help="Redis connection URL (default: from settings)",
    )
    
    args = parser.parse_args()
    
    # Get settings
    try:
        settings = get_settings()
        redis_url = args.redis_url or settings.redis_url
    except Exception as e:
        print(f"Warning: Could not load settings: {e}")
        redis_url = args.redis_url or "redis://localhost:6379"
    
    # Get username
    username = args.username
    if not username:
        username = input("Enter admin username (default: admin): ").strip() or "admin"
    
    # Get email
    email = args.email
    if not email:
        while not email or "@" not in email:
            email = input("Enter admin email: ").strip()
            if not email or "@" not in email:
                print("Please enter a valid email address")
    
    # Get password
    password = args.password
    if not password:
        while not password or len(password) < 6:
            password = getpass.getpass("Enter admin password (min 6 chars): ")
            if not password or len(password) < 6:
                print("Password must be at least 6 characters")
            else:
                password_confirm = getpass.getpass("Confirm password: ")
                if password != password_confirm:
                    print("Passwords don't match. Try again.")
                    password = None
    
    # Create admin user
    success = asyncio.run(create_admin_user(username, email, password, redis_url))
    
    if success:
        print("\n✅ Setup complete! You can now start the API server.")
        sys.exit(0)
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

