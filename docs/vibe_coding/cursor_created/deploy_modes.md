# Deployment Modes

Your 3DAIGC API supports two deployment modes to handle the multi-worker scheduler conflict.

## Mode 1: Single Worker (Simple)

**Use Case:** Development, testing, or small-scale production

**How it works:**
- FastAPI and GPU scheduler run in the same process
- No Redis required
- Singleton pattern prevents conflicts (but limits to 1 worker)

**Deployment:**
```bash
uvicorn api.main:app --workers 1 --host 0.0.0.0 --port 8000
```

**Pros:**
- ✅ Simple setup
- ✅ No external dependencies
- ✅ Good for development

**Cons:**
- ❌ Limited to single worker (no horizontal scaling)
- ❌ API and scheduler in same process

---

## Mode 2: Multi-Worker with Separate Scheduler (Production)

**Use Case:** Production environments with high traffic

**How it works:**
- Multiple FastAPI workers handle HTTP requests
- Separate scheduler service processes GPU jobs
- Redis queue coordinates between them

**Architecture:**
```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  FastAPI    │  │  FastAPI    │  │  FastAPI    │
│  Worker 1   │  │  Worker 2   │  │  Worker N   │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
                 ┌──────▼──────┐
                 │ Redis Queue │
                 └──────┬──────┘
                        │
                ┌───────▼───────┐
                │   Scheduler   │
                │   Service     │
                │ (GPU Workers) │
                └───────────────┘
```

**Deployment:**

1. **Start Redis:**
```bash
docker run -d -p 6379:6379 redis:latest
```

2. **Configure Redis in `config/system.yaml`:**
```yaml
redis_url: "redis://localhost:6379"
redis_enabled: true
```

3. **Start Scheduler Service (single instance):**
```bash
python scripts/scheduler_service.py --redis-url redis://localhost:6379
```

4. **Start FastAPI with Multiple Workers:**
```bash
uvicorn api.main_multiworker:app --workers 4 --host 0.0.0.0 --port 8000
```

**Pros:**
- ✅ True horizontal scaling
- ✅ Separate API and GPU processing
- ✅ Can scale API independently
- ✅ Production-ready

**Cons:**
- ❌ Requires Redis
- ❌ More complex setup

---

## How It Works Internally

### SchedulerAdapter (Transparency Layer)

The `api/dependencies.py` provides a `SchedulerAdapter` that makes both modes work identically from the router's perspective:

```python
# Routers use this dependency
scheduler = Depends(get_scheduler)

# All these work in both modes!
job_id = await scheduler.schedule_job(job_request)
status = await scheduler.get_job_status(job_id)
models = scheduler.get_available_models("mesh_generation")
is_valid = scheduler.validate_model_preference(model_id, feature)
```

**In Single-Worker Mode:**
- All methods call the actual scheduler directly

**In Multi-Worker Mode:**
- `scheduler.schedule_job()` → submits to Redis queue
- `scheduler.get_job_status()` → queries Redis
- `scheduler.get_available_models()` → reads from settings (same config file as scheduler service)
- `scheduler.validate_model_preference()` → validates using config from settings
- Separate scheduler service polls Redis and processes jobs

### What Works in Each Mode

| Feature | Single-Worker | Multi-Worker |
|---------|--------------|--------------|
| Submit jobs | ✅ | ✅ |
| Query job status | ✅ | ✅ |
| Cancel jobs | ✅ | ✅ |
| List available models | ✅ | ✅ (from settings) |
| Validate model preference | ✅ | ✅ (from settings) |
| Real-time GPU status | ✅ | ❌ (not available in API) |
| Worker status | ✅ | ❌ (not available in API) |
| Queue statistics | ✅ | ✅ (from Redis) |

**Note on Multi-Worker Mode:**
- Model information (available models, validation) works because API workers read the same config files as the scheduler service
- Real-time GPU status is not available via API in multi-worker mode (it's in the scheduler service)
- This is by design - API workers don't need direct GPU access

---

## Quick Start

### Development (Mode 1):
```bash
# Just run it!
uvicorn api.main:app --reload
```

### Production (Mode 2):
```bash
# Terminal 1: Redis
docker run -d -p 6379:6379 redis:latest

# Terminal 2: Scheduler Service
python scripts/scheduler_service.py

# Terminal 3: API Workers
uvicorn api.main_multiworker:app --workers 4
```

---

## Configuration

### Enable Redis Mode

Edit `config/system.yaml`:
```yaml
environment: "production"
redis_url: "redis://localhost:6379"
redis_enabled: true  # Set to true for multi-worker mode
```

Or use environment variables:
```bash
export P3D_REDIS_URL="redis://localhost:6379"
export P3D_REDIS_ENABLED="true"
```

### Important: Config Synchronization in Multi-Worker Mode

In multi-worker mode, both the **scheduler service** and **API workers** must read from the **same configuration files** (`config/models.yaml` and `config/system.yaml`).

**Why?**
- API workers need to know available models (to list them and validate model preferences)
- Scheduler service needs to know which models to load
- If configs differ, clients might see models that don't actually exist, or validation will fail incorrectly

**Best Practice:**
```bash
# Ensure both services use the same config directory
/opt/3daigc/
  ├── config/
  │   ├── system.yaml     # Shared config
  │   └── models.yaml     # Shared config
  ├── scripts/scheduler_service.py
  └── api/
```

**In Docker/Kubernetes:**
Mount the same config volume to both containers:
```yaml
volumes:
  - ./config:/app/config:ro  # Read-only
```

---

## Monitoring

### Check Queue Status

```bash
# Redis CLI
redis-cli ZCARD 3daigc:queue:pending

# API Endpoint
curl http://localhost:8000/api/v1/system/status
```

### Logs

```bash
# Scheduler service
tail -f /var/log/3daigc/scheduler.log

# API workers
tail -f /var/log/3daigc/api.log
```

---

## Which Mode Should You Use?

| Scenario | Recommended Mode |
|----------|-----------------|
| Development | Single Worker |
| Testing | Single Worker |
| Low traffic (<100 req/min) | Single Worker |
| High traffic (>100 req/min) | Multi-Worker |
| Need horizontal scaling | Multi-Worker |
| Multiple servers | Multi-Worker |

---

## Troubleshooting

### "Model scheduler is not available"
- **Single-worker mode**: Check that `api/main.py` started correctly
- **Multi-worker mode**: Check that `scheduler_service.py` is running

### Jobs not processing
- Check Redis is running: `redis-cli ping`
- Check scheduler service logs
- Verify GPU availability: `nvidia-smi`

### Multiple schedulers conflict
- You're running `api/main.py` with `--workers > 1` → Use Mode 2 instead
- Or run with `--workers 1`

