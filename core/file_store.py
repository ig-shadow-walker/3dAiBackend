"""
File metadata storage layer using Redis.
"""

import json
import logging
import time
from typing import Dict, List, Optional

from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class FileStore:
    """
    Redis-based storage for file metadata.

    Redis Key Schema:
        - file:{file_id} -> File metadata JSON (with expiration)
        - files:all -> ZSET of file_ids scored by upload timestamp
        - files:{file_type} -> ZSET of file_ids for a specific type, scored by upload timestamp
    """

    def __init__(self, redis_client: Redis, key_prefix: str = "3daigc", default_ttl_seconds: int = 86400):
        self.redis = redis_client
        self.prefix = key_prefix
        self.default_ttl = default_ttl_seconds

    def _key(self, *parts) -> str:
        """Generate Redis key with prefix"""
        return ":".join([self.prefix] + list(parts))

    async def store_file_metadata(self, file_id: str, file_info: Dict) -> bool:
        """
        Store file metadata in Redis with an expiration time.

        Args:
            file_id: Unique identifier for the file.
            file_info: Dictionary containing file metadata.

        Returns:
            True if successful, False otherwise.
        """
        try:
            file_key = self._key("file", file_id)
            all_files_key = self._key("files", "all")
            file_type_key = self._key("files", file_info.get("file_type", "unknown"))
            
            upload_timestamp = int(time.time())

            # Use a pipeline for atomic operations
            async with self.redis.pipeline() as pipe:
                pipe.set(file_key, json.dumps(file_info), ex=self.default_ttl)
                pipe.zadd(all_files_key, {file_id: upload_timestamp})
                pipe.zadd(file_type_key, {file_id: upload_timestamp})
                await pipe.execute()

            logger.info(f"Stored metadata for file {file_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store metadata for file {file_id}: {e}")
            return False

    async def get_file_metadata(self, file_id: str) -> Optional[Dict]:
        """
        Get file metadata by ID from Redis.

        Args:
            file_id: Unique identifier for the file.

        Returns:
            A dictionary with file metadata or None if not found.
        """
        try:
            file_key = self._key("file", file_id)
            metadata_json = await self.redis.get(file_key)
            if not metadata_json:
                return None
            return json.loads(metadata_json)
        except Exception as e:
            logger.error(f"Failed to get metadata for file {file_id}: {e}")
            return None

    async def delete_file_metadata(self, file_id: str) -> bool:
        """
        Delete file metadata from Redis.

        Args:
            file_id: Unique identifier for the file.

        Returns:
            True if successful, False otherwise.
        """
        try:
            metadata = await self.get_file_metadata(file_id)
            if not metadata:
                return False  # Already gone

            file_key = self._key("file", file_id)
            all_files_key = self._key("files", "all")
            file_type_key = self._key("files", metadata.get("file_type", "unknown"))

            async with self.redis.pipeline() as pipe:
                pipe.delete(file_key)
                pipe.zrem(all_files_key, file_id)
                pipe.zrem(file_type_key, file_id)
                await pipe.execute()
            
            logger.info(f"Deleted metadata for file {file_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete metadata for file {file_id}: {e}")
            return False

    async def list_file_metadata(self, file_type: Optional[str] = None, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        List file metadata, sorted by upload time.

        Args:
            file_type: Optional filter by file type (e.g., 'image', 'mesh').
            limit: Maximum number of records to return.
            offset: Starting offset for pagination.

        Returns:
            A list of file metadata dictionaries.
        """
        files = []
        try:
            if file_type:
                key = self._key("files", file_type)
            else:
                key = self._key("files", "all")

            # Get file_ids from the sorted set, newest first
            file_ids = await self.redis.zrevrange(key, offset, offset + limit - 1)

            for file_id_bytes in file_ids:
                file_id = file_id_bytes.decode('utf-8')
                metadata = await self.get_file_metadata(file_id)
                if metadata:
                    files.append(metadata)
            return files
        except Exception as e:
            logger.error(f"Failed to list file metadata: {e}")
            return files
            
    async def count_files(self, file_type: Optional[str] = None) -> int:
        """
        Count the total number of files.

        Args:
            file_type: Optional filter by file type.

        Returns:
            The total count of files.
        """
        try:
            if file_type:
                key = self._key("files", file_type)
            else:
                key = self._key("files", "all")
            
            return await self.redis.zcard(key)
        except Exception as e:
            logger.error(f"Failed to count files: {e}")
            return 0
