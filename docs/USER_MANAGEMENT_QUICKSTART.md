# User Management - Quick Start Guide

## TL;DR

User authentication is **optional** and controlled by a single flag. When disabled, the system works exactly as before.

## Configuration Flag

```bash
# Enable user authentication (job isolation)
export P3D_SECURITY__USER_AUTH_ENABLED=true

# Disable user authentication (simple mode - default)
export P3D_SECURITY__USER_AUTH_ENABLED=false
```

## Quick Setup (3 Steps)

### 1. Start Redis (if not already running)

```bash
docker run -d -p 6379:6379 redis:latest
```

### 2. Enable User Authentication

```bash
export P3D_SECURITY__USER_AUTH_ENABLED=true
```

### 3. Create Admin User

```bash
python scripts/create_admin_user.py
```

Follow the prompts to create your first admin user.

## Start the Services

```bash
# Start scheduler service
python scripts/scheduler_service.py --redis-url redis://localhost:6379 &

# Start API server with authentication enabled
P3D_SECURITY__USER_AUTH_ENABLED=true uvicorn api.main_multiworker:app --host 0.0.0.0 --port 7842 --workers 4
```

## Usage

### Register a User

```bash
curl -X POST http://localhost:7842/api/v1/users/register \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","email":"alice@example.com","password":"secret123"}'
```

### Login and Get Token

```bash
TOKEN=$(curl -s -X POST http://localhost:7842/api/v1/users/login \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"secret123"}' | jq -r '.token')

echo "Your token: $TOKEN"
```

### Use Token in Requests

```bash
# Submit a job
curl -X POST http://localhost:7842/api/v1/mesh-generation \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a 3d cat"}'

# Check job status
curl -X GET http://localhost:7842/api/v1/system/jobs/{job_id} \
  -H "Authorization: Bearer $TOKEN"

# List your jobs (only YOUR jobs will appear)
curl -X GET http://localhost:7842/api/v1/system/jobs/history \
  -H "Authorization: Bearer $TOKEN"
```

## Switching Modes

### Disable Authentication (Back to Simple Mode)

Just restart the server without the flag:

```bash
# Stop the server
# Then start without the flag or with it set to false
uvicorn api.main_multiworker:app --host 0.0.0.0 --port 7842 --workers 4
```

No authentication required. All clients see all jobs.

### Enable Authentication Again

```bash
P3D_SECURITY__USER_AUTH_ENABLED=true uvicorn api.main_multiworker:app --host 0.0.0.0 --port 7842 --workers 4
```

User isolation is enforced.

## What Changes When Authentication is Enabled?

| Feature | Simple Mode (disabled) | Authenticated Mode (enabled) |
|---------|----------------------|----------------------------|
| Job Submission | No token required | Token required |
| Job Visibility | All users see all jobs | Users only see their own jobs |
| Admin Access | N/A | Admins see all jobs |
| User Management | N/A | `/api/v1/users/*` endpoints available |
| Job Isolation | None | Enforced by `user_id` |

## Modified Files Summary

The implementation touches these key files:

### Core Authentication
- `core/auth/models.py` - User and APIToken models
- `core/auth/storage.py` - Redis-based storage
- `core/auth/service.py` - Authentication logic
- `core/config.py` - Added `user_auth_enabled` flag

### Job Ownership
- `core/scheduler/job_queue.py` - Added `user_id` field
- `core/scheduler/database_models.py` - Added `user_id` column
- `core/scheduler/redis_job_queue.py` - Serialize/deserialize `user_id`

### API Layer
- `api/dependencies.py` - Auth dependencies
- `api/routers/users.py` - User management endpoints
- `api/routers/system.py` - Job filtering by user
- `api/main_multiworker.py` - Conditional auth service initialization

### Utilities
- `scripts/create_admin_user.py` - Admin user creation script
- `docs/USER_MANAGEMENT.md` - Full documentation

## Architecture Notes

### Redis Storage Structure

#### Users
```
3daigc:user:{user_id} -> User JSON
3daigc:user:username:{username} -> user_id
3daigc:user:email:{email} -> user_id
3daigc:users:all -> Set of user_ids
```

#### Tokens
```
3daigc:token:{token} -> APIToken JSON
3daigc:user:{user_id}:tokens -> Set of token strings
```

#### Jobs (with user_id)
```
3daigc:jobs -> Hash of job_id -> job_data (includes user_id)
```

### Security Features

- **Password Hashing**: SHA-256 with random salt
- **Token Generation**: 256-bit random tokens
- **Token Expiration**: Configurable per-token
- **Role-Based Access**: USER and ADMIN roles
- **Last-Used Tracking**: Tokens track last usage

## Troubleshooting

### "Authentication service is not available"
- Ensure `user_auth_enabled: true` in config
- Check Redis is running and accessible
- Verify Redis URL in settings

### Can't see any jobs
- You're in authenticated mode and can only see YOUR jobs
- Check your token with: `GET /api/v1/users/me`
- Admins should see all jobs

### Want to disable authentication
- Simply restart server with flag set to `false`
- Existing jobs remain in database
- Jobs without `user_id` are visible to everyone in simple mode

## Need More Help?

See the full documentation: `docs/USER_MANAGEMENT.md`

