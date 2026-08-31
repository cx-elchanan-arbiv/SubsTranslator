FROM python:3.12-slim

# Set an environment variable to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including ffmpeg and Hebrew fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    fonts-liberation \
    fonts-dejavu-core \
    fonts-noto \
    fonts-noto-cjk \
    fonts-noto-hinted \
    fonts-noto-unhinted \
    wget \
 && rm -rf /var/lib/apt/lists/*

# deno: a JavaScript runtime for yt-dlp's benefit, not ours. YouTube's newer
# protections are small JS challenges; yt-dlp can solve them but needs a JS
# engine on the machine, and it already WARNS on every run that extraction
# without one is deprecated ("some formats may be missing"). Installing deno
# now is insurance against the next YouTube flip — the third one this month,
# after the stale-pin 403s and the android_vr 403s. yt-dlp detects it on PATH
# automatically; nothing else needs configuring.
RUN ARCH=$(uname -m) && \
    case "$ARCH" in \
      x86_64)  DENO_ARCH=x86_64-unknown-linux-gnu ;; \
      aarch64) DENO_ARCH=aarch64-unknown-linux-gnu ;; \
      *) echo "unsupported arch $ARCH" && exit 1 ;; \
    esac && \
    wget -q "https://github.com/denoland/deno/releases/latest/download/deno-${DENO_ARCH}.zip" -O /tmp/deno.zip && \
    apt-get update && apt-get install -y --no-install-recommends unzip && \
    unzip -q /tmp/deno.zip -d /usr/local/bin && \
    chmod 755 /usr/local/bin/deno && \
    rm /tmp/deno.zip && \
    apt-get purge -y unzip && rm -rf /var/lib/apt/lists/* && \
    deno --version

# Create non-root user for security (Celery warns when running as root)
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Download and install Hebrew fonts
RUN mkdir -p /usr/share/fonts/truetype/hebrew && \
    # Download Noto Sans Hebrew
    wget -q "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansHebrew/NotoSansHebrew-Regular.ttf" \
         -O /usr/share/fonts/truetype/hebrew/NotoSansHebrew-Regular.ttf && \
    wget -q "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansHebrew/NotoSansHebrew-Bold.ttf" \
         -O /usr/share/fonts/truetype/hebrew/NotoSansHebrew-Bold.ttf && \
    # Update font cache
    fc-cache -fv

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a dedicated directory for yt-dlp cache with correct permissions
RUN mkdir -p /app/yt_dlp_cache && chmod 777 /app/yt_dlp_cache
ENV XDG_CACHE_HOME=/app/yt_dlp_cache

# Copy application code
# This should be one of the last steps to leverage Docker layer caching
COPY backend/ .

# Make start script executable
RUN chmod +x /app/start.sh

# Create necessary directories and set ownership to non-root user
RUN mkdir -p /app/uploads /app/downloads /app/whisper_models /app/fast_work /app/assets && \
    chown -R appuser:appgroup /app && \
    chmod 755 /app/uploads /app/downloads /app/fast_work

# Render injects $PORT; keep a sane default for local runs
ENV PORT=10000
EXPOSE ${PORT}

# Switch to non-root user (eliminates Celery root warning)
USER appuser

# Command to run the application using start.sh
# start.sh runs both Gunicorn and Celery worker
CMD ["/app/start.sh"]
