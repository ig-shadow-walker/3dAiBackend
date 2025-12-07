"""FastAPI dependencies for dependency injection"""

import logging
from typing import Any, Dict, Optional

from core.config import get_settings
from core.scheduler.job_queue import JobRequest
from fastapi import Depends, Header, HTTPException, Request

logger = logging.getLogger(__name__)


class SchedulerAdapter:
    """
    Adapter that provides a consistent scheduler interface for both deployment modes.
    
    In single-worker mode: wraps the actual scheduler
    In multi-worker mode: wraps the Redis job queue and loads model info from settings
    """
    
    def __init__(self, scheduler=None, job_queue=None, settings=None):
        self._scheduler = scheduler
        self._job_queue = job_queue
        self._settings = settings or get_settings()
        self._mode = "single_worker" if scheduler else "multi_worker"
        
        # In multi-worker mode, build model registry from settings
        self._model_features = {}  # feature -> [model_ids]
        self._model_registry = {}  # model_id -> config
        
        if not scheduler and settings:
            self._load_model_info_from_settings()
    
    def _load_model_info_from_settings(self):
        """Load model information from settings (for multi-worker mode)"""
        try:
            from core.scheduler.model_factory import get_model_configs_from_settings
            
            model_configs = get_model_configs_from_settings(self._settings.models)
            
            for model_id, config in model_configs.items():
                self._model_registry[model_id] = config
                feature_type = config.get("feature_type")
                
                if feature_type:
                    if feature_type not in self._model_features:
                        self._model_features[feature_type] = []
                    self._model_features[feature_type].append(model_id)
            
            logger.info(f"Loaded {len(model_configs)} models from settings for multi-worker mode")
        except Exception as e:
            logger.error(f"Failed to load model info from settings: {e}")
    
    async def schedule_job(self, job_request: JobRequest) -> str:
        """Schedule a job (compatible with both modes)"""
        if self._scheduler:
            return await self._scheduler.schedule_job(job_request)
        else:
            return await self._job_queue.enqueue(job_request)
    
    async def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get job status (compatible with both modes)"""
        if self._scheduler:
            return await self._scheduler.get_job_status(job_id)
        else:
            return await self._job_queue.get_job(job_id)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job (compatible with both modes)"""
        if self._scheduler:
            return await self._scheduler.cancel_job(job_id)
        else:
            return await self._job_queue.cancel_job(job_id)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        if self._scheduler:
            return await self._scheduler.get_system_status()
        else:
            # In multi-worker mode, return what we can from Redis and settings
            queue_status = await self._job_queue.get_queue_status()
            return {
                "scheduler": {
                    "type": "external_service",
                    "mode": "multi_worker",
                    "note": "Scheduler runs as separate service. GPU status not available via API.",
                    "num_registered_models": len(self._model_registry),
                },
                "queue": queue_status,
                "features": {
                    feature: len(models) 
                    for feature, models in self._model_features.items()
                },
                "gpu": {
                    "note": "GPU status only available from scheduler service",
                    "available": None
                }
            }
    
    def validate_model_preference(self, model_id: str, feature: str) -> bool:
        """Validate model preference"""
        if self._scheduler:
            return self._scheduler.validate_model_preference(model_id, feature)
        else:
            # In multi-worker mode, validate using our loaded model registry
            if not model_id:
                return True  # No preference is always valid
            
            # Check if model exists in registry
            if model_id not in self._model_registry:
                return False
            
            # Check if model supports the requested feature
            if feature not in self._model_features:
                return False
            
            return model_id in self._model_features[feature]
    
    def get_available_models(self, feature: Optional[str] = None) -> Dict[str, Any]:
        """Get available models"""
        if self._scheduler:
            return self._scheduler.get_available_models(feature)
        else:
            # In multi-worker mode, return from our loaded model registry
            if feature:
                return {feature: self._model_features.get(feature, [])}
            return dict(self._model_features)
    
    @property
    def mode(self) -> str:
        """Get deployment mode"""
        return self._mode
    
    @property
    def job_queue(self):
        """
        Get the underlying job queue.
        
        For compatibility with code that directly accesses scheduler.job_queue
        """
        if self._scheduler:
            return self._scheduler.job_queue
        else:
            return self._job_queue
    
    @property
    def job_queue(self):
        """Get the underlying job queue (for direct access when needed)"""
        if self._scheduler:
            return self._scheduler.job_queue
        else:
            return self._job_queue

async def get_current_settings():
    """Get current application settings"""
    return get_settings()

async def get_scheduler(request: Request):
    """
    Get a scheduler interface that works in both deployment modes.
    
    Returns a SchedulerAdapter that provides consistent methods whether running in:
    - Single-worker mode (scheduler embedded in FastAPI)
    - Multi-worker mode (separate scheduler service with Redis queue)
    
    In multi-worker mode, the adapter loads model information from settings,
    so clients can still query available models and validate model preferences.
    """
    settings = get_settings()
    
    # Try single-worker mode first
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is not None:
        return SchedulerAdapter(scheduler=scheduler, settings=settings)
    
    # Fall back to multi-worker mode
    job_queue = getattr(request.app.state, "job_queue", None)
    if job_queue is not None:
        return SchedulerAdapter(job_queue=job_queue, settings=settings)
    
    # Neither available - service not configured properly
    raise HTTPException(
        status_code=503,
        detail="Scheduler service is not available. Please check the service configuration."
    )


async def get_job_queue(request: Request):
    """
    Get the job queue instance from app state.
    
    This works in both modes:
    - Single-worker mode: Returns scheduler.job_queue
    - Multi-worker mode: Returns the Redis job queue directly
    """
    # Try to get scheduler first (single-worker mode)
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is not None:
        return scheduler.job_queue
    
    # Fall back to direct job_queue (multi-worker mode)
    job_queue = getattr(request.app.state, "job_queue", None)
    if job_queue is None:
        raise HTTPException(
            status_code=503,
            detail="Job queue is not available. Please try again later."
        )
    return job_queue


async def get_scheduler_or_queue(request: Request):
    """
    Get either scheduler or job queue, whichever is available.
    
    Returns a dict with 'mode', 'scheduler', and 'job_queue' keys.
    Use this for endpoints that need to work in both deployment modes.
    """
    scheduler = getattr(request.app.state, "scheduler", None)
    job_queue = getattr(request.app.state, "job_queue", None)
    
    if scheduler is not None:
        # Single-worker mode
        return {
            "mode": "single_worker",
            "scheduler": scheduler,
            "job_queue": scheduler.job_queue
        }
    elif job_queue is not None:
        # Multi-worker mode
        return {
            "mode": "multi_worker", 
            "scheduler": None,
            "job_queue": job_queue
        }
    else:
        raise HTTPException(
            status_code=503,
            detail="Neither scheduler nor job queue is available. Please check the service configuration."
        )

async def verify_api_key(
    authorization: Optional[str] = Header(None),
    settings = Depends(get_current_settings)
):
    """Verify API key if authentication is enabled"""
    if not settings.security.api_key_required:
        return True
    
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing"
        )
    
    # Basic API key validation (you can enhance this)
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format. Use 'Bearer <token>'"
        )
    
    # For now, just check if token is present
    # You can implement proper JWT or API key validation here
    token = authorization.split(" ")[1]
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )
    
    return True

async def check_rate_limit(
    request_ip: str = None,
    settings = Depends(get_current_settings)
):
    """Check rate limiting (basic implementation)"""
    # This is a placeholder for rate limiting
    # In production, you'd use Redis or similar for distributed rate limiting
    return True


async def get_file_store(request: Request):
    """
    Get the FileStore instance from app state.
    
    Returns the Redis-backed FileStore in multi-worker mode, or None in single-worker mode.
    The file_upload router handles both cases - using Redis when available, falling back
    to in-memory storage for single-worker mode.
    
    Returns:
        FileStore instance or None
    """
    return getattr(request.app.state, "file_store", None)

class CommonQueryParams:
    """Common query parameters for API endpoints"""
    def __init__(
        self,
        skip: int = 0,
        limit: int = 100,
        sort_by: Optional[str] = None,
        order: str = "asc"
    ):
        self.skip = skip
        self.limit = min(limit, 1000)  # Cap at 1000
        self.sort_by = sort_by
        self.order = order


# User authentication dependencies

async def get_auth_service(request: Request):
    """
    Get the authentication service from app state.
    
    The auth service is initialized during application startup and stored in app.state.
    """
    settings = get_settings()
    
    if not settings.user_auth_enabled:
        raise HTTPException(
            status_code=503,
            detail="User authentication is disabled on this server. Enable it in configuration."
        )
    
    auth_service = getattr(request.app.state, "auth_service", None)
    if auth_service is None:
        raise HTTPException(
            status_code=503,
            detail="Authentication service is not available. Please check the service configuration."
        )
    return auth_service


async def get_current_user(
    authorization: Optional[str] = Header(None),
    request: Request = None,
):
    """
    Get the current authenticated user from the Authorization header.
    
    This dependency validates the Bearer token and returns the associated User object.
    All authenticated endpoints should use this dependency.
    
    If user_auth_enabled is False, this will raise an error.
    
    Usage:
        @router.get("/protected")
        async def protected_endpoint(user: User = Depends(get_current_user)):
            return {"user_id": user.user_id}
    """
    from core.auth.models import User
    
    settings = get_settings()
    
    # Check if user auth is enabled
    if not settings.user_auth_enabled:
        raise HTTPException(
            status_code=503,
            detail="User authentication is disabled on this server."
        )
    
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing. Please provide a Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format. Use 'Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = authorization.split(" ")[1]
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get auth service and validate token
    auth_service = await get_auth_service(request)
    user, error = await auth_service.validate_token(token)
    
    if error or not user:
        raise HTTPException(
            status_code=401,
            detail=error or "Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


async def get_current_user_optional(
    authorization: Optional[str] = Header(None),
    request: Request = None,
):
    """
    Get the current authenticated user, but return None if not authenticated or if auth is disabled.
    
    This is useful for endpoints that work with or without authentication,
    but provide additional features for authenticated users.
    
    If user_auth_enabled is False, always returns None (no authentication).
    """
    settings = get_settings()
    
    # If user auth is disabled, return None (no filtering)
    if not settings.user_auth_enabled:
        return None
    
    if not authorization:
        return None
    
    try:
        return await get_current_user(authorization, request)
    except HTTPException:
        return None


async def require_admin(current_user = Depends(get_current_user)):
    """
    Require that the current user has admin role.
    
    Usage:
        @router.get("/admin-only")
        async def admin_endpoint(_: None = Depends(require_admin)):
            return {"message": "Admin access granted"}
    """
    from core.auth.models import UserRole
    
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    
    return current_user


async def get_current_user_or_none(
    authorization: Optional[str] = Header(None),
    request: Request = None,
):
    """
    Smart dependency for job submission endpoints.
    
    Behavior depends on user_auth_enabled setting:
    - If user_auth_enabled=True: REQUIRES authentication (returns User or raises 401)
    - If user_auth_enabled=False: Authentication optional (returns User or None)
    
    This ensures that when user management is enabled, all job submissions
    require authentication. When disabled, the system runs in simple mode.
    
    Usage in job submission endpoints:
        @router.post("/some-job")
        async def submit_job(
            current_user = Depends(get_current_user_or_none),
            ...
        ):
            # current_user will be:
            # - User object if authenticated
            # - None if auth is disabled
            # - Raises 401 if auth is enabled but no valid token provided
            
            user_id = current_user.user_id if current_user else None
            job_request = JobRequest(..., user_id=user_id)
    """
    settings = get_settings()
    
    if settings.user_auth_enabled:
        # Authentication is REQUIRED when user_auth_enabled=True
        return await get_current_user(authorization, request)
    else:
        # Authentication is OPTIONAL when user_auth_enabled=False (simple mode)
        return await get_current_user_optional(authorization, request)
