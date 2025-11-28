# Deployment Strategies for Multi-Worker Environments

## Problem Overview

The GPU scheduler manages shared resources (GPUs, worker processes) and needs to coordinate job processing. When running FastAPI with multiple uvicorn workers, each worker process creates its own scheduler instance, causing:

- **Resource Conflicts**: Multiple schedulers competing for GPU allocation
- **Duplicate Job Queues**: Each scheduler maintains its own queue
- **Worker Process Duplication**: Multiple sets of GPU workers
- **Inconsistent State**: Job status varies across different API workers

## Solution Architectures

### Option 1: Separate Scheduler Service (★ RECOMMENDED for Production)

**Architecture:**
```
┌─────────────────────────────────────────────────────────┐
│                      Load Balancer                       │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌───────▼────────┐  ┌──────▼─────────┐
│  FastAPI       │  │  FastAPI       │  │  FastAPI       │
│  Worker 1      │  │  Worker 2      │  │  Worker N      │
│  (API only)    │  │  (API only)    │  │  (API only)    │
└───────┬────────┘  └───────┬────────┘  └───────┬────────┘
        │                   │                    │
        └───────────────────┼────────────────────┘
                            │
                   ┌────────▼─────────┐
                   │   Redis Queue    │
                   └────────┬─────────┘
                            │
                   ┌────────▼─────────┐
                   │   Scheduler      │
                   │   Service        │
                   │  (GPU Workers)   │
                   └──────────────────┘
```

**Implementation:**

1. **Start Redis:**
```bash
docker run -d -p 6379:6379 redis:latest
```

2. **Start Scheduler Service (Single Instance):**
```bash
python scripts/scheduler_service.py --redis-url redis://localhost:6379
```

3. **Start FastAPI with Multiple Workers:**
```bash
uvicorn api.main_multiworker:app --host 0.0.0.0 --port 8000 --workers 4
```

**Pros:**
- ✅ True horizontal scaling of API layer
- ✅ Single source of truth for GPU resource management
- ✅ Easy to monitor and restart scheduler independently
- ✅ Can run scheduler on GPU-equipped machine, API on separate machines
- ✅ Best for production deployments

**Cons:**
- ❌ Requires Redis infrastructure
- ❌ Slightly more complex deployment
- ❌ Network latency for job submission (minimal)

**Use Cases:**
- Production environments with high request volume
- Distributed deployments (API servers separate from GPU servers)
- When you need true horizontal scaling

---

### Option 2: Single Uvicorn Worker (★ SIMPLEST)

**Architecture:**
```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI + Scheduler                   │
│                     (Single Process)                     │
│                                                          │
│  ┌──────────────┐      ┌──────────────────────────┐   │
│  │  FastAPI     │──────│  GPU Scheduler           │   │
│  │  (asyncio)   │      │  (multiprocess workers)  │   │
│  └──────────────┘      └──────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**Implementation:**

Use existing `api/main.py` with single worker:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

**Pros:**
- ✅ No code changes required
- ✅ Simplest deployment
- ✅ No external dependencies (Redis not needed)
- ✅ Good for development and testing

**Cons:**
- ❌ Limited FastAPI concurrency (only asyncio, no process parallelism)
- ❌ Single point of failure
- ❌ Cannot scale API horizontally

**Use Cases:**
- Development and testing
- Low to medium traffic deployments
- Single-server setups with sufficient asyncio concurrency

---

### Option 3: Hybrid with Process Coordination

**Architecture:**
```
┌─────────────────────────────────────────────────────────┐
│                      Load Balancer                       │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌───────▼────────┐  ┌──────▼─────────┐
│  FastAPI       │  │  FastAPI       │  │  FastAPI       │
│  Worker 1      │  │  Worker 2      │  │  Worker N      │
│ (queue only)   │  │ (queue only)   │  │ (scheduler)    │
└───────┬────────┘  └───────┬────────┘  └───────┬────────┘
        │                   │                    │
        └───────────────────┼────────────────────┘
                            │
                   ┌────────▼─────────┐
                   │   Redis Queue    │
                   └──────────────────┘
```

**Implementation:**

Designate one uvicorn worker as the "scheduler worker" using environment variables:

```python
# In main.py
import os

enable_scheduler = os.getenv("ENABLE_SCHEDULER", "false").lower() == "true"

scheduler = MultiprocessModelScheduler(
    gpu_monitor=gpu_monitor,
    job_queue=redis_queue,
    enable_processing=enable_scheduler,  # Only true for one worker
)
```

**Start:**
```bash
# Worker 1 (with scheduler)
ENABLE_SCHEDULER=true uvicorn api.main:app --port 8001

# Workers 2-N (API only)
ENABLE_SCHEDULER=false uvicorn api.main:app --port 8002 &
ENABLE_SCHEDULER=false uvicorn api.main:app --port 8003 &
```

**Pros:**
- ✅ Scales API layer
- ✅ Single codebase
- ✅ Scheduler runs within FastAPI ecosystem

**Cons:**
- ❌ Manual coordination of which worker runs scheduler
- ❌ More complex process management
- ❌ Harder to restart scheduler independently

---

## Comparison Matrix

| Feature | Separate Service | Single Worker | Hybrid |
|---------|-----------------|---------------|---------|
| Horizontal Scaling | ✅ Excellent | ❌ No | ✅ Good |
| Deployment Complexity | ⚠️ Medium | ✅ Simple | ⚠️ Medium |
| Resource Isolation | ✅ Excellent | ❌ No | ⚠️ Partial |
| External Dependencies | Redis | None | Redis |
| Production Ready | ✅ Yes | ⚠️ Limited | ⚠️ Yes |
| Development Friendly | ⚠️ Medium | ✅ Excellent | ⚠️ Medium |

---

## Recommended Approach by Environment

### Development
```bash
# Use single worker (Option 2)
uvicorn api.main:app --reload --workers 1
```

### Staging
```bash
# Use separate service (Option 1) to test production setup
docker-compose up  # See docker-compose.yml
```

### Production
```bash
# Use separate service (Option 1)
# With process manager (systemd, supervisord, kubernetes)

# Service 1: Scheduler
systemctl start scheduler-service

# Service 2: FastAPI (multiple workers)
gunicorn api.main_multiworker:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

---

## Migration Guide

### From Single Worker to Separate Service

1. **Add Redis configuration to `core/config.py`:**
```python
class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"
```

2. **Update routers to use Redis job queue:**
```python
# In your routers
job_queue = request.app.state.job_queue  # This will be RedisJobQueue
job_id = await job_queue.enqueue(job_request)
```

3. **Deploy:**
```bash
# Terminal 1: Start Redis
docker run -d -p 6379:6379 redis:latest

# Terminal 2: Start scheduler service
python scripts/scheduler_service.py

# Terminal 3: Start API with multiple workers
uvicorn api.main_multiworker:app --workers 4
```

---

## Monitoring and Operations

### Health Checks

**Scheduler Service:**
```bash
# Check if scheduler is processing jobs
curl http://localhost:8000/api/v1/system/status
```

**Redis Queue:**
```bash
# Check queue depth
redis-cli ZCARD 3daigc:queue:pending
```

### Logging

All services log to stdout. Use process managers to aggregate:

```bash
# systemd
journalctl -u scheduler-service -f

# Docker Compose
docker-compose logs -f scheduler
```

### Troubleshooting

**Problem: Jobs not being processed**
- Check scheduler service is running: `ps aux | grep scheduler_service`
- Check Redis connectivity: `redis-cli ping`
- Check GPU availability: `nvidia-smi`

**Problem: Jobs stuck in queue**
- Check scheduler logs for errors
- Verify VRAM availability
- Check worker process health: `ps aux | grep model_worker`

---

## Performance Tuning

### FastAPI Workers
```bash
# Calculate optimal workers: (2 x CPU cores) + 1
uvicorn api.main_multiworker:app --workers $((2 * $(nproc) + 1))
```

### Redis Configuration
```bash
# Increase max memory for large job queues
redis-cli CONFIG SET maxmemory 2gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

### GPU Scheduler
```python
# In model configs, adjust max_workers per model
{
    "model_id": "high_demand_model",
    "max_workers": 3,  # Allow 3 concurrent instances
}
```

---

## Docker Compose Example

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  scheduler:
    build: .
    command: python scripts/scheduler_service.py --redis-url redis://redis:6379
    depends_on:
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  api:
    build: .
    command: uvicorn api.main_multiworker:app --host 0.0.0.0 --port 8000 --workers 4
    ports:
      - "8000:8000"
    depends_on:
      - redis
      - scheduler
    environment:
      - REDIS_URL=redis://redis:6379

volumes:
  redis_data:
```

Start everything:
```bash
docker-compose up --scale api=3
```

This gives you:
- 1 Redis instance
- 1 Scheduler service
- 3 API containers (each with 4 workers = 12 total FastAPI workers)

---

## Conclusion

For **production deployments**, use **Option 1 (Separate Scheduler Service)** with Redis as the job queue. This provides:
- True horizontal scalability
- Clean separation of concerns
- Easy monitoring and maintenance
- Production-grade reliability

For **development**, use **Option 2 (Single Worker)** for simplicity.

