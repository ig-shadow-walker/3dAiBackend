# User Management System

The 3DAIGC API supports optional user management and authentication, allowing you to control who can access jobs and ensure users can only see their own work.

## Table of Contents

- [Overview](#overview)
- [Enabling/Disabling User Authentication](#enablingdisabling-user-authentication)
- [Creating the First Admin User](#creating-the-first-admin-user)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [User Roles](#user-roles)
- [Job Isolation](#job-isolation)

## Overview

The user management system provides:

- **Token-based authentication**: Secure API tokens for client authentication
- **Role-based access control**: USER and ADMIN roles
- **Job isolation**: Users can only see/access their own jobs
- **Redis-based storage**: No additional database required
- **Optional**: Can be completely disabled for simple deployments

## Enabling/Disabling User Authentication

User authentication is controlled by a configuration flag that can be set in multiple ways:

### Method 1: Environment Variable

```bash
export P3D_SECURITY__USER_AUTH_ENABLED=true
# or
export P3D_SECURITY__USER_AUTH_ENABLED=false
```

### Method 2: Configuration File

Edit your `config.yaml`:

```yaml
security:
  user_auth_enabled: true  # or false
  api_key_required: false
  cors_origins: ["*"]
```

### Method 3: Command Line (when starting server)

```bash
# Enable user auth
P3D_SECURITY__USER_AUTH_ENABLED=true uvicorn api.main_multiworker:app --host 0.0.0.0 --port 7842

# Disable user auth (simple mode)
P3D_SECURITY__USER_AUTH_ENABLED=false uvicorn api.main_multiworker:app --host 0.0.0.0 --port 7842
```

## Behavior Modes

### 🔒 Authenticated Mode (`user_auth_enabled: true`)

- Users must register and login to get API tokens
- Each job is associated with the user who submitted it
- Users can **only** see and access their own jobs
- Admin users can see and access **all** jobs
- User management endpoints are available

### 🌐 Simple Mode (`user_auth_enabled: false`)

- No authentication required
- All clients can see all jobs
- No user management
- Suitable for trusted environments or development

## Creating the First Admin User

When you first enable user authentication, you need to create an admin user:

```bash
# Interactive mode
python scripts/create_admin_user.py

# Non-interactive mode
python scripts/create_admin_user.py \
  --username admin \
  --email admin@example.com \
  --password your_secure_password \
  --redis-url redis://localhost:6379
```

This will create an admin user and generate an API token for immediate use.

## API Endpoints

### User Registration and Authentication

#### Register a new user

```bash
POST /api/v1/users/register
Content-Type: application/json

{
  "username": "john",
  "email": "john@example.com",
  "password": "secret123"
}
```

Response:
```json
{
  "success": true,
  "message": "User registered successfully. Please login to get an API token.",
  "user": {
    "user_id": "user_abc123",
    "username": "john",
    "email": "john@example.com",
    "role": "user",
    "is_active": true,
    "created_at": "2024-01-01T00:00:00"
  }
}
```

#### Login and get API token

```bash
POST /api/v1/users/login
Content-Type: application/json

{
  "username": "john",
  "password": "secret123"
}
```

Response:
```json
{
  "user": {...},
  "token": "abc123xyz789...",
  "token_name": "Login token - john",
  "message": "Login successful. Use this token in Authorization header."
}
```

### User Profile Management

#### Get current user profile

```bash
GET /api/v1/users/me
Authorization: Bearer <your_token>
```

#### Change password

```bash
PUT /api/v1/users/me/password
Authorization: Bearer <your_token>
Content-Type: application/json

{
  "old_password": "secret123",
  "new_password": "newsecret456"
}
```

### Token Management

#### Create a new API token

```bash
POST /api/v1/users/tokens
Authorization: Bearer <your_token>
Content-Type: application/json

{
  "token_name": "My App Token",
  "expires_in_days": 365
}
```

#### List your tokens

```bash
GET /api/v1/users/tokens
Authorization: Bearer <your_token>
```

#### Revoke a token

```bash
DELETE /api/v1/users/tokens/{token_or_prefix}
Authorization: Bearer <your_token>
```

### Admin Endpoints

#### List all users (admin only)

```bash
GET /api/v1/users/admin/users?limit=100&offset=0
Authorization: Bearer <admin_token>
```

#### Get user by ID (admin only)

```bash
GET /api/v1/users/admin/users/{user_id}
Authorization: Bearer <admin_token>
```

#### Delete user (admin only)

```bash
DELETE /api/v1/users/admin/users/{user_id}
Authorization: Bearer <admin_token>
```

## Usage Examples

### Full Workflow Example

```bash
# 1. Register a new user
curl -X POST http://localhost:7842/api/v1/users/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "alice",
    "email": "alice@example.com",
    "password": "mypassword"
  }'

# 2. Login to get API token
TOKEN=$(curl -X POST http://localhost:7842/api/v1/users/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "alice",
    "password": "mypassword"
  }' | jq -r '.token')

# 3. Submit a job (automatically associated with your user)
JOB_ID=$(curl -X POST http://localhost:7842/api/v1/mesh-generation \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a 3d cat model",
    "model_preference": "trellis"
  }' | jq -r '.job_id')

# 4. Check job status (only YOUR jobs are visible)
curl -X GET http://localhost:7842/api/v1/system/jobs/$JOB_ID \
  -H "Authorization: Bearer $TOKEN"

# 5. List your job history (only YOUR jobs)
curl -X GET http://localhost:7842/api/v1/system/jobs/history \
  -H "Authorization: Bearer $TOKEN"

# 6. Download result when complete
curl -X GET http://localhost:7842/api/v1/system/jobs/$JOB_ID/download \
  -H "Authorization: Bearer $TOKEN" \
  -o result.glb
```

### Admin Workflow Example

```bash
# Login as admin
ADMIN_TOKEN=$(curl -X POST http://localhost:7842/api/v1/users/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "admin_password"
  }' | jq -r '.token')

# List all users
curl -X GET http://localhost:7842/api/v1/users/admin/users \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# View all jobs (admin sees everything)
curl -X GET http://localhost:7842/api/v1/system/jobs/history \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Delete a user
curl -X DELETE http://localhost:7842/api/v1/users/admin/users/user_abc123 \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

## User Roles

### USER Role (Default)

- Can register and login
- Can submit jobs
- Can **only** view their own jobs
- Can manage their own tokens
- Cannot access other users' jobs
- Cannot manage other users

### ADMIN Role

- All USER permissions
- Can view **all** jobs from **all** users
- Can list all users
- Can view user details
- Can delete users
- Full system access

## Job Isolation

When user authentication is **enabled**:

### Job Submission
- Each job is automatically tagged with the `user_id` of the submitter
- No manual user_id needed - it's extracted from the authentication token

### Job Queries
- `GET /api/v1/system/jobs/{job_id}` - Returns 403 if job doesn't belong to you
- `GET /api/v1/system/jobs/history` - Only returns YOUR jobs
- `GET /api/v1/system/jobs/queue/stats` - Shows queue stats (no isolation)
- `GET /api/v1/system/jobs/{job_id}/download` - Only download YOUR jobs

### Admin Override
- Admin users bypass all filters
- Admins can view/access/delete any job
- Useful for support and monitoring

## Security Considerations

1. **Password Storage**: Passwords are hashed using SHA-256 with random salt
2. **Token Security**: API tokens are long random strings (256-bit)
3. **Token Expiration**: Optional expiration for tokens
4. **HTTPS**: Always use HTTPS in production
5. **Redis Security**: Secure your Redis instance with password and firewall

## Migration from Simple Mode

If you're currently running without authentication and want to enable it:

1. **Enable the flag**: Set `user_auth_enabled: true`
2. **Create admin user**: Run the `create_admin_user.py` script
3. **Restart server**: Restart with the new configuration
4. **Notify users**: Users need to register and get tokens
5. **Existing jobs**: Jobs without `user_id` will only be visible to admins

## Troubleshooting

### "Authentication service is not available"

- Ensure Redis is running and accessible
- Check that `user_auth_enabled: true` in configuration
- Verify Redis URL is correct in settings

### "Access denied to this job"

- You're trying to access a job that doesn't belong to you
- Only the job owner or an admin can access it
- Check you're using the correct token

### Can't see any jobs

- In authenticated mode, you only see YOUR jobs
- If you're an admin, you should see all jobs
- Check your token is valid with `GET /api/v1/users/me`

## FAQ

**Q: Can I switch between modes without losing data?**  
A: Yes! Job data is preserved. However, jobs without `user_id` won't be visible to regular users when you enable auth.

**Q: Can I use both username and API token?**  
A: No, you must use API tokens in the Authorization header. Username/password is only for getting tokens.

**Q: How many tokens can a user have?**  
A: Unlimited. Users can create multiple tokens for different applications.

**Q: Can I customize token expiration?**  
A: Yes, when creating a token, specify `expires_in_days` parameter.

**Q: What happens if I forget the admin password?**  
A: You can reset it via Redis directly or create a new admin user with a script.

