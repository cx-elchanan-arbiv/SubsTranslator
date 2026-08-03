# SubsTranslator - Operations & Runbook

## Quick Start (Local Development)

### Prerequisites

```bash
# Required Software
- Docker Desktop 4.0+ with Compose V2
- Python 3.12+ (for local backend dev)
- Node.js 20+ with npm 10+ (for frontend dev)
- FFmpeg 4.4+ (or use containerized version)

# Verify installations
docker --version && docker compose version
python3 --version && node --version && npm --version
ffmpeg -version || echo "FFmpeg will use containerized version"
```

### Environment Setup

1. **Copy environment template**:
```bash
cp .env.example .env
# Edit .env with your actual values
```

2. **Required .env configuration**:
```bash
FLASK_ENV=development
DEBUG=True
OPENAI_API_KEY=your-openai-api-key-here
```

### One-liner Startup

```bash
# Full stack (recommended)
./scripts/start.sh

# Or manual Docker Compose
docker compose up --build

# Frontend dev server (hot reloading)
cd frontend && npm start

# Backend dev server (with auto-reload)
cd backend && FLASK_ENV=development python app.py
```

### Smoke Test Checklist

```bash
# ✅ Check all services running
docker compose ps
# Expected: 5 services (frontend, backend, worker, beat, redis) - all "Up"

# ✅ Test endpoints
curl http://localhost/                     # Frontend (200 OK)
curl http://localhost:8081/health          # Backend health (200 OK)
curl http://localhost:8081/languages       # API data (200 JSON)

# ✅ Check Redis connection
docker compose exec redis redis-cli ping   # Should return "PONG"

# ✅ Test file upload limit
curl -X POST -F "file=@small_test.mp4" http://localhost:8081/upload
# Should return task_id or validation error (not connection error)
```

## Environment Variables

### Required Variables

| Variable | Purpose | Local Example | CI/Prod Example |
|----------|---------|---------------|-----------------|
| `FLASK_ENV` | Flask runtime mode | `development` | `production` |
| `DEBUG` | Debug logging enable | `True` | `False` |
| `OPENAI_API_KEY` | OpenAI translation service | `sk-proj-...` | `${{ secrets.OPENAI_API_KEY }}` |
| `DEFAULT_WHISPER_MODEL` | Initial model size | `tiny` | `medium` |
| `WORKER_CONCURRENCY` | Celery worker count | `1` | `4` |
| `MAX_FILE_SIZE` | Upload limit (bytes) | `500000000` | `1000000000` |
| `LOG_LEVEL` | Logging verbosity | `DEBUG` | `WARNING` |

### Optional Variables

| Variable | Purpose | Default | Notes |
|----------|---------|---------|-------|
| `REDIS_HOST` | Redis server | `redis` | Docker service name |
| `TASK_SOFT_TIME_LIMIT` | Worker timeout (sec) | `1800` | 30 minutes |
| `REQUIRE_DOWNLOAD_TOKEN` | Secure downloads | `False` | Set `True` in prod |

### Secret Management

**Local Development**:
```bash
# .env file (never commit)
echo ".env" >> .gitignore
cp .env.example .env
# Edit .env with real values
```

**CI/CD (GitHub Actions)**:
```yaml
# .github/workflows/ci.yml
env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  
# Repository Settings → Secrets → Actions
# Add: OPENAI_API_KEY = sk-proj-...
```

**Key Rotation Procedure**:
1. Generate new key at https://platform.openai.com/api-keys
2. Update GitHub repository secrets
3. Update local `.env` files
4. Revoke old key after verification

## Common Operations

### Service Management

```bash
# Start all services
./scripts/start.sh
docker compose up -d --build

# Stop gracefully
./scripts/stop.sh
docker compose down

# Restart single service
docker compose restart backend
docker compose restart worker

# Scale workers (during high load)
docker compose up -d --scale worker=3

# View service status
docker compose ps
docker stats
```

### Queue Management

```bash
# Check Redis queue status
docker compose exec redis redis-cli
> LLEN processing        # Queue depth
> KEYS celery*          # All Celery keys
> FLUSHDB               # Clear all queues (emergency)

# Monitor worker activity
docker compose logs -f worker
docker compose exec worker celery -A celery_worker.celery_app inspect stats

# Purge failed tasks
docker compose exec worker celery -A celery_worker.celery_app purge
```

### File Management

```bash
# Check storage usage
du -sh backend/uploads backend/downloads backend/whisper_models

# Manual cleanup (if automated cleanup fails)
find backend/uploads -type f -mtime +1 -delete
find backend/downloads -type f -mtime +1 -delete

# Refresh Whisper models cache
rm -rf backend/whisper_models/*
# Models will re-download automatically on next transcription

# Manual retention policy enforcement
python scripts/cleanup_expired_files.py --dry-run
python scripts/cleanup_expired_files.py --force
```

### Task Management

```bash
# Manual task retry (with actual task ID)
TASK_ID="abc123-def456-789"
curl -X POST "http://localhost:8081/retry/${TASK_ID}"

# Check specific task status
curl "http://localhost:8081/status/${TASK_ID}" | jq

# Kill stuck task
docker compose exec worker celery -A celery_worker.celery_app control revoke ${TASK_ID} --terminate

# List active tasks
docker compose exec worker celery -A celery_worker.celery_app inspect active
```

## Observability & Troubleshooting

### Log Locations & Patterns

```bash
# Backend application logs
docker compose logs -f backend | grep -E "(ERROR|WARNING|task_)"

# Worker processing logs
docker compose logs -f worker | grep -E "(transcription|download|error)"

# Frontend access logs (Nginx)
docker compose logs -f frontend | grep -v "GET /static"

# Redis operations
docker compose logs -f redis | grep -E "(DENIED|ERROR)"

# Filter by request ID (structured logging)
docker compose logs backend | grep "request_id=req_abc123"

# Error aggregation
docker compose logs --since=1h | grep -E "ERROR|CRITICAL|Exception" | sort | uniq -c
```

### Metrics & Monitoring

**Current Metrics** (Basic):
```bash
# Health endpoint
curl http://localhost:8081/health | jq
# Returns: {"status": "healthy", "timestamp": "...", "services": {...}}

# Task queue metrics
curl http://localhost:8081/metrics | jq
# TODO: Implement Prometheus metrics endpoint
```

**Recommended Monitoring Setup**:
```python
# Add to app.py for production
from prometheus_client import Counter, Histogram, Gauge

task_counter = Counter('tasks_total', 'Total tasks', ['status', 'type'])
processing_time = Histogram('task_duration_seconds', 'Task duration')
queue_depth = Gauge('queue_depth', 'Current queue depth')
```

**Request Correlation**:
```bash
# Add request ID to all logs
X-Request-ID: req_$(uuidgen)
# Track across services: frontend → backend → worker
```

### Common Failure Playbooks

#### yt-dlp Download Failures

**Symptoms**: Tasks stuck in download step, "Video unavailable" errors
```bash
# Check yt-dlp version and cache
docker compose exec worker yt-dlp --version
docker compose exec worker ls -la /tmp/yt-dlp/

# Clear yt-dlp cache
docker compose exec worker rm -rf /tmp/yt-dlp/*

# Test URL manually
docker compose exec worker yt-dlp --simulate "https://youtu.be/VIDEO_ID"

# Update yt-dlp (if needed)
docker compose exec worker pip install --upgrade yt-dlp
docker compose restart worker
```

#### FFmpeg Processing Failures

**Symptoms**: Audio extraction or video muxing failures
```bash
# Check FFmpeg installation
docker compose exec worker ffmpeg -version

# Test audio extraction manually
docker compose exec worker ffmpeg -i /app/uploads/test.mp4 -vn -acodec pcm_s16le test.wav

# Common codec issues
# H.264 not available: install x264
# Audio codec issues: check input format

# Resource issues
docker stats worker  # Check memory/CPU usage
# If OOM: increase memory limits in docker-compose.yml
```

#### Whisper Model Issues

**Symptoms**: Transcription timeouts, OOM errors, model loading failures
```bash
# Check available models
ls -la backend/whisper_models/

# Test model loading
docker compose exec worker python -c "
from whisper_smart import smart_whisper
model = smart_whisper('tiny')
print('Model loaded successfully')
"

# Memory issues: fallback to smaller model
# Timeout issues: increase TASK_SOFT_TIME_LIMIT
# Network issues: check Hugging Face connectivity
```

#### Upload Validation Failures

**Symptoms**: 400 errors on file upload, "Invalid file type"
```bash
# Check file validation rules
grep -n "ALLOWED_EXTENSIONS" backend/config.py

# Test upload manually
curl -X POST -F "file=@test.mp4" \
     -F "source_lang=auto" -F "target_lang=he" \
     http://localhost:8081/upload

# Common issues:
# - File size > MAX_FILE_SIZE
# - Unsupported format (check ALLOWED_EXTENSIONS)
# - Filename with special characters
```

### Diagnostic Collection

**Bug Report Checklist**:
```bash
# 1. System info
docker compose version
docker compose ps
docker stats --no-stream

# 2. Service logs (last 100 lines)
docker compose logs --tail=100 backend > logs_backend.txt
docker compose logs --tail=100 worker > logs_worker.txt

# 3. Configuration
env | grep -E "(FLASK|CELERY|OPENAI)" | sed 's/sk-[^=]*/sk-***REDACTED***/g'

# 4. Queue state
docker compose exec redis redis-cli INFO | grep -E "(memory|keyspace)"
docker compose exec redis redis-cli LLEN processing

# 5. File system state
du -sh backend/{uploads,downloads,whisper_models}
ls -la backend/downloads/ | head -10
```

## Testing & CI

### Local Testing

```bash
# Unit tests (fast, no external dependencies)
python -m pytest tests/ -m "unit" -v

# Integration tests (requires running services)
./scripts/start.sh  # Start services first
python -m pytest tests/ -m "integration" -v

# E2E tests (full flow with real files)
python -m pytest tests/ -m "e2e" -v --tb=short

# Specific test categories
python -m pytest tests/ -m "not slow" -v           # Skip long-running tests
python -m pytest tests/ -k "youtube" -v            # YouTube-related tests only
python -m pytest tests/ -k "hebrew" -v             # RTL/i18n tests only

# Frontend tests
cd frontend
npm test                    # Jest unit tests
npm run test:e2e           # Playwright E2E tests
npm run test:localization  # i18n validation tests
```

### CI Pipeline Overview

**Jobs** (`.github/workflows/ci.yml`):
1. **backend-tests**: Python unit/integration tests with coverage
2. **frontend-tests**: React/TypeScript tests with coverage  
3. **lint-and-format**: Code quality (black, ruff, ESLint)
4. **security-check**: Dependency audit + static analysis

**Caching Strategy**:
- Python dependencies: `~/.cache/pip`
- Node modules: `frontend/node_modules`
- Whisper models: Not cached (too large)

**Artifacts**:
- Coverage reports → Codecov
- Test results → GitHub Actions summary
- Security scan results → Sarif upload

### CI Environment Variables

```yaml
# Required secrets in GitHub repository settings
secrets:
  OPENAI_API_KEY: "sk-proj-..."          # OpenAI API access
  CODECOV_TOKEN: "..."                   # Coverage reporting

# Public environment variables
env:
  FLASK_ENV: testing
  DEFAULT_WHISPER_MODEL: tiny            # Faster for CI
  MAX_FILE_SIZE: 10485760               # 10MB limit for tests
  USE_FAKE_YTDLP: true                  # Mock YouTube downloads
```

### Test Data & Mocking

```bash
# Test assets location
tests/assets/test_video.mp4              # Sample video file
tests/manual/test_download_only_quick.py # Manual testing scripts

# Mock services in CI
USE_FAKE_YTDLP=true        # Skip real YouTube downloads
OPENAI_API_KEY=sk-test     # Fake key for translation tests
```

## Release & Versioning

### Current State
- **No formal versioning** (technical debt)
- **Manual releases** via Docker builds
- **No changelog** automation

### Recommended Process

```bash
# 1. Version bump (semantic versioning)
echo "1.2.3" > VERSION
git tag v1.2.3

# 2. Generate changelog
git log --oneline $(git describe --tags --abbrev=0)..HEAD > CHANGELOG_DRAFT.md

# 3. Build release artifacts
docker compose build --no-cache
docker tag substranslator_backend:latest substranslator_backend:v1.2.3

# 4. Health verification
./scripts/verify_substranslator.sh

# 5. Deploy (manual for now)
docker compose up -d
# TODO: Add automated deployment pipeline
```

### Rollback Procedure

```bash
# 1. Stop current services
docker compose down

# 2. Revert to previous version
git checkout v1.2.2
docker compose up --build -d

# 3. Verify health
curl http://localhost:8081/health
# Check processing pipeline with test video

# 4. Clear any corrupted state
docker compose exec redis redis-cli FLUSHDB
```

## Security Hygiene

### Secret Management

**Local Development**:
```bash
# ✅ DO: Use .env files (gitignored)
cp .env.example .env
echo ".env" >> .gitignore

# ❌ DON'T: Commit real keys
git diff --staged | grep -E "sk-|api[_-]?key" && echo "🚨 SECRET DETECTED"
```

**Pre-commit Checks**:
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Manual secret scan
grep -r "sk-proj-\|sk-[a-zA-Z0-9]" . --exclude-dir=.git || echo "✅ No secrets found"

# CI secret prevention
git log --grep="sk-" --oneline | head -5  # Check git history
```

### Upload Security Validation

```bash
# File type allowlist (config.py:30)
ALLOWED_EXTENSIONS = {"mp4", "mkv", "mov", "webm", "avi", "mp3", "wav"}

# Size limits enforcement
MAX_FILE_SIZE = 500MB  # Configurable per environment

# Path traversal prevention
secure_filename()  # Werkzeug utility used throughout

# Content validation (TODO)
# Implement MIME type checking beyond file extensions
```

### Rate Limiting Configuration

```python
# TODO: Add Flask-Limiter
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour", "20 per minute"]
)

# Apply to endpoints
@app.route('/youtube', methods=['POST'])
@limiter.limit("5 per minute")  # Heavy processing
def youtube_process():
    pass
```

## Common Operations

### Service Control

```bash
# Graceful restart (preserves queue state)
docker compose restart backend worker

# Force restart (clears memory, loses in-progress tasks)
docker compose down && docker compose up -d

# Scale workers for high load
docker compose up -d --scale worker=4

# Update single service
docker compose build backend
docker compose up -d --no-deps backend
```

### Queue Operations

```bash
# Clear all queues (emergency reset)
docker compose exec redis redis-cli FLUSHDB

# Clear specific queue
docker compose exec worker celery -A celery_worker.celery_app purge -Q processing

# Inspect queue contents
docker compose exec worker celery -A celery_worker.celery_app inspect scheduled
docker compose exec worker celery -A celery_worker.celery_app inspect active

# Cancel all tasks
docker compose exec worker celery -A celery_worker.celery_app control revoke --terminate
```

### Model Management

```bash
# Check model cache status
ls -lah backend/whisper_models/
du -sh backend/whisper_models/

# Force model re-download
rm -rf backend/whisper_models/models--Systran--faster-whisper-*
# Models auto-download on next use

# Test model loading
docker compose exec worker python -c "
from whisper_smart import smart_whisper
for model in ['tiny', 'base', 'medium']:
    print(f'Testing {model}...')
    smart_whisper(model)
    print(f'✅ {model} loaded successfully')
"
```

### Data Cleanup

```bash
# Manual cleanup (bypass scheduled cleanup)
docker compose exec worker python -c "
from tasks import cleanup_files_task
result = cleanup_files_task.delay()
print(f'Cleanup task: {result.id}')
"

# Emergency disk space recovery
docker system prune -f
docker volume prune -f

# Check largest files
find backend/ -type f -size +100M -exec ls -lh {} \; | sort -k5 -hr
```

## Observability & Troubleshooting

### Log Analysis

```bash
# Error pattern matching
docker compose logs | grep -E "ERROR|Exception|Traceback" | tail -20

# Performance monitoring
docker compose logs worker | grep -E "Task.*completed|processing_time" | tail -10

# Request flow tracing (if request_id implemented)
REQUEST_ID="req_abc123"
docker compose logs | grep $REQUEST_ID | sort

# Task lifecycle tracking
TASK_ID="task_xyz789"
docker compose logs | grep $TASK_ID | grep -E "(started|progress|completed|failed)"
```

### Performance Diagnostics

```bash
# Resource utilization
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

# Processing bottlenecks
# Check queue depth vs worker capacity
PROCESSING_QUEUE=$(docker compose exec redis redis-cli LLEN processing)
ACTIVE_WORKERS=$(docker compose exec worker celery -A celery_worker.celery_app inspect stats | jq '.[] | length')
echo "Queue: $PROCESSING_QUEUE, Workers: $ACTIVE_WORKERS"

# Identify slow operations
docker compose logs worker | grep "processing_time" | awk '{print $NF}' | sort -n | tail -5
```

### Health Monitoring

```bash
# Comprehensive health check
./scripts/verify_substranslator.sh

# Individual service health
curl http://localhost:8081/health | jq '.services'
# Expected: {"redis": "connected", "disk": "ok", "memory": "ok"}

# Worker health (Celery inspect)
docker compose exec worker celery -A celery_worker.celery_app inspect ping
# Expected: {"worker@hostname": "pong"}

# Database connectivity (Redis)
docker compose exec redis redis-cli ping
docker compose exec redis redis-cli INFO replication
```

### Common Error Resolution

#### "Worker not responding"
```bash
# Check worker process
docker compose ps worker
docker compose logs --tail=50 worker

# Restart worker
docker compose restart worker

# Check for memory issues
docker stats worker --no-stream
# If memory > 7GB: restart worker, consider scaling down concurrent tasks
```

#### "Task timeout errors"
```bash
# Check timeout configuration
grep -n "TIME_LIMIT" backend/config.py

# Increase timeout for large files
export TASK_SOFT_TIME_LIMIT=3600  # 1 hour
docker compose restart worker

# Monitor long-running tasks
docker compose exec worker celery -A celery_worker.celery_app inspect active | jq '.[] | .[] | .time_start'
```

#### "Translation service errors"
```bash
# Test OpenAI API connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models | jq '.data | length'

# Fallback to Google Translate
export DEFAULT_TRANSLATION_SERVICE=google
docker compose restart backend

# Check quota/rate limits
docker compose logs backend | grep -E "quota|rate.limit|429"
```

## Testing & CI

### Test Execution Commands

```bash
# Quick test suite (< 2 minutes)
python -m pytest tests/ -m "unit and not slow" -v

# Full test suite (< 10 minutes)
python -m pytest tests/ -v --tb=short

# Coverage report
python -m pytest tests/ --cov=backend --cov-report=html --cov-report=term

# Frontend tests
cd frontend
npm test -- --coverage --watchAll=false
npm run test:e2e  # Playwright tests
```

### CI Debugging

```bash
# Reproduce CI environment locally
export FLASK_ENV=testing
export USE_FAKE_YTDLP=true
export OPENAI_API_KEY=sk-test-fake

# Run CI test suite
python -m pytest tests/ -m "not integration and not e2e" -v --tb=short

# Check CI artifacts
# Download from GitHub Actions → Artifacts → coverage-report
```

### Test Data Management

```bash
# Test video location
tests/assets/test_video.mp4  # 5MB sample file

# Generate test data
ffmpeg -f lavfi -i testsrc=duration=10:size=320x240:rate=1 -pix_fmt yuv420p tests/assets/test_short.mp4

# Mock YouTube responses (CI)
tests/fixtures/youtube_metadata.json    # Fake video metadata
tests/fixtures/fake_video.mp4          # Placeholder download
```

## Appendix

### Directory Map (Operations Relevant)

```
SubsTranslator/
├── 📊 Monitoring & Logs
│   ├── backend/logs/                # Application logs (if file logging enabled)
│   └── frontend/build/static/       # Static assets + source maps
│
├── 💾 Runtime Data  
│   ├── backend/uploads/             # Temporary input files
│   ├── backend/downloads/           # Output artifacts (24h retention)
│   ├── backend/whisper_models/      # ML model cache (persistent)
│   └── backend/assets/              # Static resources (logos, fonts)
│
├── 🛠️ Operations Scripts
│   ├── scripts/start.sh             # Development startup
│   ├── scripts/stop.sh              # Graceful shutdown  
│   ├── scripts/dev.sh               # Development mode
│   ├── scripts/clean_safe_data.sh   # Safe cleanup
│   └── scripts/verify_substranslator.sh  # Health verification
│
└── 🔧 Configuration
    ├── .env                         # Local environment (gitignored)
    ├── .env.example                 # Template for setup
    ├── docker-compose.yml           # Service definitions
    └── nginx.conf                   # Reverse proxy config
```

### Command Cheat Sheet

```bash
# Quick Operations
./scripts/start.sh                          # Start all services
./scripts/stop.sh                           # Stop all services  
docker compose restart worker               # Restart worker only
docker compose logs -f worker | head -50   # Live worker logs

# Diagnostics
curl http://localhost:8081/health | jq     # Backend health
docker stats --no-stream                   # Resource usage
docker compose exec redis redis-cli ping   # Redis connectivity

# Queue Management
docker compose exec redis redis-cli LLEN processing        # Queue depth
docker compose exec worker celery -A celery_worker.celery_app inspect active  # Active tasks

# Emergency Recovery
docker compose down && docker compose up -d --build        # Full restart
docker compose exec redis redis-cli FLUSHDB               # Clear all queues
find backend/uploads -mtime +0 -delete                    # Force file cleanup
```

### FAQ

**Q: What ports are used?**
A: Frontend (80), Backend (8081), Redis (6379)

**Q: How long do processing jobs take?**
A: 2-15 minutes for 10-minute videos, depending on model size and content

**Q: What's the maximum file size?**
A: 500MB default, configurable via `MAX_FILE_SIZE` environment variable

**Q: How to increase worker concurrency?**
A: Set `WORKER_CONCURRENCY=4` in .env, restart worker service

**Q: Where are processed files stored?**
A: `backend/downloads/` with 24-hour retention via automated cleanup

**Q: How to enable GPU acceleration?**
A: Add GPU support to Docker Compose + set `WHISPER_DEVICE=cuda` in config

**Q: What happens if Redis goes down?**
A: All tasks fail gracefully, restart Redis to resume processing

**Q: How to monitor queue depth?**
A: `docker compose exec redis redis-cli LLEN processing`

---

*Last Updated: 2025-08-31*  
*Document Version: 1.0*  
*For: Development & Operations Teams*