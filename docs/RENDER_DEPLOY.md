# 🚀 Render.com Deployment Guide - Consolidated Deploy

**Goal:** Get Backend (with integrated Celery Worker) running on Render.com
**Architecture:** Single service running Gunicorn + Celery Worker together
**Time:** ~20 minutes
**Cost:** ~$21-25/month (Standard service + disks)

---

## 📋 Prerequisites Checklist

Before starting, ensure you have:

- [ ] GitHub repository pushed with latest changes
- [ ] Upstash Redis instance created
- [ ] Upstash credentials rotated (see Step 0 below)
- [ ] OpenAI API key
- [ ] YouTube API key (optional, but recommended)

---

## 🏗️ Architecture Overview

**Important Change (Nov 17, 2025):** We've consolidated the worker into the backend service!

### Why One Service?
Previously, we had 2 separate services:
- Backend (Flask/Gunicorn)
- Worker (Celery)

This caused sync issues and complexity. Now we run **both in one service** using `backend/start.sh`:

```bash
📡 Gunicorn (web server) - port 10000
⚙️  Celery Worker (background tasks) - concurrent
```

### Benefits:
- ✅ Simpler deployment
- ✅ Lower cost (~$21 vs ~$35/month)
- ✅ No sync issues between services
- ✅ Easier to manage

### Trade-offs:
- ⚠️ Single point of failure (but Render auto-restarts)
- ⚠️ Scaling requires scaling both together

---

## 🔐 Step 0: Rotate Upstash Credentials (CRITICAL!)

Since the password was exposed in chat, **rotate it first**:

1. Go to [Upstash Console](https://console.upstash.io/)
2. Select your Redis database: `complete-oriole-9147`
3. Click **Details** → **Reset Password** (or rotate credentials)
4. Copy the new connection string (should look like: `rediss://default:NEW_PASSWORD@complete-oriole-9147.upstash.io:6379`)
5. **Save this string** - you'll need it for Step 2!

---

## 🎯 Step 1: Create Backend Service (Web)

### 1.1 Create New Web Service

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **New +** → **Web Service**
3. Connect your GitHub repository: `SubsTranslator`

### 1.2 Configure Service

Fill in these **exact** values:

| Field | Value |
|-------|-------|
| **Name** | `substranslator-backend` |
| **Region** | Click "Deploy in a new region" → **Europe (Frankfurt)** |
| **Branch** | `main` |
| **Root Directory** | (leave empty) |
| **Environment** | **Docker** |
| **Dockerfile Path** | `./backend.Dockerfile` |
| **Instance Type** | **Standard** (1 CPU / 2 GB RAM / $25/month) |

### 1.3 Add Environment Variables

Click **Advanced** → **Add Environment Variable** and add these one by one:

```bash
# Redis/Celery (use YOUR new Upstash password!)
REDIS_URL=rediss://default:YOUR_NEW_PASSWORD@complete-oriole-9147.upstash.io:6379
CELERY_BROKER_URL=rediss://default:YOUR_NEW_PASSWORD@complete-oriole-9147.upstash.io:6379/0
CELERY_RESULT_BACKEND=rediss://default:YOUR_NEW_PASSWORD@complete-oriole-9147.upstash.io:6379/0
CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP=1

# Flask/App
FLASK_ENV=production
SECRET_KEY=spzBOQ2IBEFmz79jKQsG5FiPap1T3czjbUk6_v6bD6E
CORS_ORIGINS=*

# Whisper/FFmpeg/Paths
DEFAULT_WHISPER_MODEL=base
WHISPER_DEVICE=cpu
WHISPER_MODELS_FOLDER=/app/whisper_models
FAST_WORK_DIR=/app/fast_work
UPLOAD_FOLDER=/app/uploads
DOWNLOADS_FOLDER=/app/downloads
ASSETS_FOLDER=/app/assets
STATS_FOLDER=/app/stats
MAX_FILE_SIZE=524288000

# Logging
LOG_LEVEL=INFO
DEBUG=False
WORKER_PREFETCH_MULTIPLIER=1

# API Keys (replace with your actual keys!)
OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
YOUTUBE_API_KEY=YOUR_YOUTUBE_API_KEY_HERE
```

> **Note:** For `SECRET_KEY`, you can use the one provided or generate your own with:
> `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`

### 1.4 Click "Create Web Service"

Render will now:
- Clone your repository
- Build the Docker image (takes 5-10 minutes first time)
- Deploy the service

**Wait for the build to complete** before proceeding to Step 1.5.

### 1.5 Add Persistent Disks

After the service is created (can be during or after first deploy):

#### Disk 1: Whisper Models Cache

1. Go to your service → **Settings** → **Disks**
2. Click **Add Disk**
3. Configure:
   - **Name:** `whisper-cache`
   - **Mount Path:** `/app/whisper_models`
   - **Size:** `10 GB`
4. Click **Save**
5. Render will ask to redeploy → **Confirm**

#### Disk 2: Statistics Storage (IMPORTANT!)

1. Click **Add Disk** again
2. Configure:
   - **Name:** `stats-storage`
   - **Mount Path:** `/app/stats`
   - **Size:** `1 GB` (will grow slowly, 1GB = ~500K video entries)
3. Click **Save**
4. Render will ask to redeploy → **Confirm**

**Why you need this:**
All video processing statistics are saved to `/app/stats/video_stats.jsonl` for long-term analysis. Without this disk, you'll lose all historical data when the service restarts!

### 1.6 Configure Health Check

1. Go to **Settings** → **Health & Alerts**
2. Set **Health Check Path:** `/healthz`
3. Click **Save Changes**

---

## ✅ Step 2: Verify Deployment

### 2.1 Check Service Health

1. Go to your **substranslator-backend** service
2. Wait for deployment to complete (green "Live" badge)
3. Click on **Logs** tab
4. You should see both:
   ```
   📡 Starting Gunicorn...
   ⚙️  Starting Celery Worker...
   ✅ Both processes started!
   ```

### 2.2 Test the Backend

Visit your service URL (something like: `https://substranslator-backend.onrender.com`)

You should see the Flask API response.

### 2.3 Check Celery Worker

In the logs, look for:
```
[INFO] celery@... ready.
```

This confirms the worker is running and connected to Redis.

### 2.4 Troubleshooting

**If Gunicorn fails:**
- Check `PORT` environment variable is set
- Check all required env vars from Step 1.3

**If Celery Worker fails:**
- Check `CELERY_BROKER_URL` (Redis connection)
- Check Redis credentials are correct
- Look for connection errors in logs

---

## ✅ Step 3: Verify Deployment (Smoke Test)

### 3.1 Check Backend Health

1. Go to your **Backend service** in Render
2. Copy the service URL (e.g., `https://substranslator-backend.onrender.com`)
3. Open in browser or use curl:

```bash
curl -I https://YOUR-BACKEND-URL.onrender.com/healthz
```

**Expected:** `HTTP/2 200` with JSON response `{"ok": true}` or similar.

### 3.2 Check Worker Logs

1. Go to your **Worker service** in Render
2. Click **Logs** tab
3. Look for lines like:

```
celery@... ready.
Connected to redis://...
```

### 3.3 Check Upstash Redis

1. Go to [Upstash Console](https://console.upstash.io/)
2. Select your database
3. Check **Metrics** - you should see:
   - Active connections: 2-3
   - Commands/sec: occasional activity

### 3.4 Test with Simple Task (Optional but Recommended)

If your backend has a test endpoint, try submitting a small task:

```bash
# Example: Submit a short YouTube URL
curl -X POST https://YOUR-BACKEND-URL.onrender.com/api/youtube \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "target_language": "he"
  }'
```

Watch the **Worker logs** to see task processing in real-time.

---

## 🎉 Definition of Done

Your smoke deploy is **successful** when:

- [x] Backend `/healthz` returns **200 OK**
- [x] Worker logs show **"Connected to Redis / ready"**
- [x] Upstash shows active connections
- [x] (Optional) Simple task completes end-to-end

---

## 🐛 Common Issues & Fixes

### Issue: Health Check Failing

**Symptom:** Backend shows "Health check failed" in Render
**Fix:**
1. Check logs for port binding issues
2. Verify `PORT` environment variable is being used
3. Ensure `/healthz` endpoint exists in your code

### Issue: Worker Not Connecting to Redis

**Symptom:** Worker logs show "Connection refused" or "NOAUTH"
**Fix:**
1. Verify all 3 Redis URLs are correct (with **new** Upstash password)
2. Ensure URLs start with `rediss://` (with double 's' for TLS)
3. Check Upstash firewall settings (should allow all IPs)

### Issue: Build Taking Too Long

**Symptom:** Docker build exceeds 15 minutes
**Fix:**
1. First deploy always takes longer (downloading models, fonts, etc.)
2. Subsequent deploys will be faster due to layer caching
3. Consider upgrading to larger instance temporarily for faster builds

### Issue: Out of Memory (OOM)

**Symptom:** Worker crashes during task processing
**Fix:**
1. Upgrade Worker to **Standard** instance (2GB RAM)
2. Reduce `WORKER_CONCURRENCY` to `1`
3. Use `tiny` or `base` model instead of `large`

---

## 📊 Cost Breakdown (Monthly)

| Component | Details | Cost |
|---------|----------|------|
| Backend (with integrated worker) | Standard (1 CPU / 2GB) | $21 |
| Disk: whisper-cache | 10GB persistent | $0.25 |
| Disk: stats-storage | 1GB persistent | $0.25 |
| **Total** | | **~$21.50/month** |

### Cost Comparison

**Before (separate services):**
- Backend: $25/month
- Worker: $7/month
- Disks: $0.50/month
- **Total: ~$32.50/month**

**After (consolidated):**
- Backend + Worker: $21/month
- Disks: $0.50/month
- **Total: ~$21.50/month**

**Savings: $11/month (34% reduction!)** 💰

> **Note:** First month you get $50 credit, so this is essentially free to test!

---

## 🚀 Next Steps (After Smoke Deploy)

Once your smoke deploy is successful:

1. **Add Beat Service** for scheduled cleanup tasks
2. **Configure R2/S3** for persistent file storage
3. **Set up custom domain** (e.g., `api.subs.sayai.io`)
4. **Deploy Frontend** as Static Site on Render
5. **Add monitoring** (Render metrics, Sentry, etc.)
6. **Optimize worker concurrency** based on actual load

---

## 🆘 Support

- **Render Docs:** https://render.com/docs
- **Upstash Docs:** https://docs.upstash.com/redis
- **Celery Docs:** https://docs.celeryq.dev/

---

## 📝 Changelog

- **2025-01-05:** Initial smoke deploy guide created
- **2025-01-05:** Fixed backend.Dockerfile for dynamic PORT support

---

**Ready to deploy?** Start with **Step 0** and work through sequentially! 🎯
