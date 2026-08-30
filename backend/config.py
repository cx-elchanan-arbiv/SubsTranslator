"""
Configuration settings for SubsTranslator
"""

import os


class Config:
    """Base configuration class with all default settings"""

    # Application Settings
    APP_NAME = "SubsTranslator"
    APP_VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    # Server Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8081))
    WORKERS = int(os.getenv("WORKERS", 1))

    # Directory Configuration
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "/app/uploads")
    DOWNLOADS_FOLDER = os.getenv("DOWNLOADS_FOLDER", "/app/downloads")
    WHISPER_MODELS_FOLDER = os.getenv("WHISPER_MODELS_FOLDER", "/app/whisper_models")
    ASSETS_FOLDER = os.getenv("ASSETS_FOLDER", "/app/assets")

    # Phase A: Fast workspace for I/O operations
    FAST_WORK_DIR = os.getenv("FAST_WORK_DIR", "/app/fast_work")

    # Phase A+ Hotfix: OpenAI batch limits and concurrency
    MAX_TOKENS_PER_BATCH = int(os.getenv("MAX_TOKENS_PER_BATCH", 5000))
    MAX_SEGMENTS_PER_BATCH = int(
        os.getenv("MAX_SEGMENTS_PER_BATCH", 20)
    )  # Reduced from 25 to 20 to prevent merge issues
    MAX_CONCURRENT_OPENAI_REQUESTS = int(
        os.getenv("MAX_CONCURRENT_OPENAI_REQUESTS", 1)
    )  # per process
    OPENAI_REQUEST_TIMEOUT_S = int(
        os.getenv("OPENAI_REQUEST_TIMEOUT_S", 120)
    )  # Increased to 2 minutes
    MAX_OPENAI_RETRIES = int(
        os.getenv("MAX_OPENAI_RETRIES", 5)
    )  # Increased retries from 3 to 5
    ALLOW_GOOGLE_FALLBACK = (
        os.getenv("ALLOW_GOOGLE_FALLBACK", "False").lower() == "true"
    )  # Disabled by default

    # Strict Quality Gates (Production-Grade)
    ENABLE_SRT_EXPORT_GATE = (
        os.getenv("ENABLE_SRT_EXPORT_GATE", "True").lower() == "true"
    )  # Strict validation before export
    ALLOW_TRANSLATION_FALLBACK = (
        os.getenv("ALLOW_TRANSLATION_FALLBACK", "False").lower() == "true"
    )  # No fallback to original text
    GATE_MAX_RETRIES = int(os.getenv("GATE_MAX_RETRIES", 1))  # Single automatic retry
    FAIL_FAST_ON_GATE_ERROR = (
        os.getenv("FAIL_FAST_ON_GATE_ERROR", "True").lower() == "true"
    )  # Don't export on validation failure

    # Quality Gate Thresholds
    MAX_CPS = float(
        os.getenv("MAX_CPS", 22.0)
    )  # Characters per second (Hebrew: ~18-22 optimal)
    MAX_CUE_DURATION_S = float(
        os.getenv("MAX_CUE_DURATION_S", 6.0)
    )  # Max subtitle display time
    MIN_CUE_GAP_MS = float(
        os.getenv("MIN_CUE_GAP_MS", 50.0)
    )  # Min gap between cues to prevent overlap

    # File Processing Limits
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 500 * 1024 * 1024))  # 500MB default
    MAX_FILE_AGE = int(os.getenv("MAX_FILE_AGE", 24 * 3600))  # 24 hours default
    ALLOWED_EXTENSIONS: set[str] = {"mp4", "mkv", "mov", "webm", "avi", "mp3", "wav"}

    # Whisper Model Configuration
    DEFAULT_WHISPER_MODEL = os.getenv(
        "DEFAULT_WHISPER_MODEL", "large"
    )  # The owner runs locally with plenty of RAM and always wants best accuracy.
    # A constrained deployment (the old 2GB Render worker) should override via env.
    AVAILABLE_WHISPER_MODELS: set[str] = {
        "base",
        "medium",
        "large",
        "gemini",
    }  # Added gemini as experimental option

    # Hosted Mode Configuration
    # When True: Restricts resource-intensive models (large) to PRO users only
    # When False (self-hosted): All models are available
    HOSTED_MODE = os.getenv("HOSTED_MODE", "False").lower() == "true"
    # Models that require PRO subscription in hosted mode
    PRO_ONLY_MODELS: set[str] = {"large"}
    WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")

    # Language Configuration
    DEFAULT_SOURCE_LANG = os.getenv("DEFAULT_SOURCE_LANG", "auto")
    # Import shared language configuration
    from shared_config import DEFAULT_TARGET_LANGUAGE as DEFAULT_TARGET_LANG
    from shared_config import LEGACY_SUPPORTED_LANGUAGES as SUPPORTED_LANGUAGES

    # Video Processing Configuration
    DEFAULT_LOGO_SIZE = int(os.getenv("DEFAULT_LOGO_SIZE", 80))  # Height in pixels
    DEFAULT_WATERMARK_OPACITY = float(os.getenv("DEFAULT_WATERMARK_OPACITY", 0.4))
    DEFAULT_WATERMARK_POSITION = ("upper_right", "comfortable")

    # Watermark paths
    WATERMARK_PATHS: dict[str, str] = {
        "default": os.path.join("/app/assets", "logo.png"),
        "alternative": os.path.join("/app/assets", "watermark.png"),
    }

    # FFmpeg Configuration
    VIDEO_QUALITY = os.getenv("VIDEO_QUALITY", "medium")
    SUBTITLE_FONT_SIZE = int(os.getenv("SUBTITLE_FONT_SIZE", 18))

    # Hebrew/RTL Font Configuration
    HEBREW_FONTS = [
        "Noto Sans Hebrew",
        "DejaVu Sans",
        "Liberation Sans",
        "Arial Hebrew Scholar",
        "Arial Hebrew",
        "David",
        "Arial Unicode MS",
    ]

    # Redis Configuration (for Celery and Rate Limiting)
    # Check for full REDIS_URL first (e.g., from Render/Upstash), then construct from parts
    REDIS_URL = os.getenv("REDIS_URL")
    if not REDIS_URL:
        REDIS_HOST = os.getenv("REDIS_HOST", "redis")
        REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
        REDIS_DB = int(os.getenv("REDIS_DB", 0))
        REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

    # API Keys & Services
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # Gemini Configuration
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GEMINI_MAX_DURATION = int(os.getenv("GEMINI_MAX_DURATION", 900))  # 15 minutes
    GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "base")
    GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", 120))  # 2 minutes

    # API Keys & Services
    # Celery Configuration
    # For TLS Redis (rediss://), append ssl_cert_reqs to URL
    # This is required because broker_use_ssl config is not reliably applied
    # See: https://github.com/celery/celery/discussions/8335
    _celery_broker = os.getenv("CELERY_BROKER_URL", REDIS_URL)
    _celery_backend = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

    # Add SSL cert requirement for TLS connections
    if (
        _celery_broker
        and _celery_broker.startswith("rediss://")
        and "ssl_cert_reqs" not in _celery_broker
    ):
        _celery_broker = f"{_celery_broker}?ssl_cert_reqs=CERT_REQUIRED"
    if (
        _celery_backend
        and _celery_backend.startswith("rediss://")
        and "ssl_cert_reqs" not in _celery_backend
    ):
        _celery_backend = f"{_celery_backend}?ssl_cert_reqs=CERT_REQUIRED"

    CELERY_BROKER_URL = _celery_broker
    CELERY_RESULT_BACKEND = _celery_backend
    CELERY_TASK_SERIALIZER = os.getenv("CELERY_TASK_SERIALIZER", "json")
    CELERY_RESULT_SERIALIZER = os.getenv("CELERY_RESULT_SERIALIZER", "json")
    CELERY_TIMEZONE = os.getenv("CELERY_TIMEZONE", "UTC")
    CELERY_ENABLE_UTC = os.getenv("CELERY_ENABLE_UTC", "True").lower() == "true"
    CELERY_RESULT_EXPIRES = int(os.getenv("CELERY_RESULT_EXPIRES", 3600))  # 1 hour

    # Task timeout settings (in seconds)
    TASK_SOFT_TIME_LIMIT = int(os.getenv("TASK_SOFT_TIME_LIMIT", 1800))  # 30 minutes
    TASK_TIME_LIMIT = int(os.getenv("TASK_TIME_LIMIT", 2100))  # 35 minutes

    # Worker settings
    WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", 2))
    WORKER_PREFETCH_MULTIPLIER = int(os.getenv("WORKER_PREFETCH_MULTIPLIER", 1))
    WORKER_MAX_TASKS_PER_CHILD = int(os.getenv("WORKER_MAX_TASKS_PER_CHILD", 100))

    # YouTube Download Configuration
    YOUTUBE_TIMEOUT = int(os.getenv("YOUTUBE_TIMEOUT", 60))  # seconds
    YOUTUBE_RETRIES = int(os.getenv("YOUTUBE_RETRIES", 3))
    YOUTUBE_FRAGMENT_RETRIES = int(os.getenv("YOUTUBE_FRAGMENT_RETRIES", 3))

    # yt-dlp player client(s).
    #
    # This used to be pinned to android_vr: YouTube forced SABR streaming on the
    # web/android clients, which stripped every adaptive format and left only
    # format 18 (360p). See https://github.com/yt-dlp/yt-dlp/issues/12482
    #
    # That workaround has since inverted. android_vr's https formats now require a
    # GVS PO token, and without one YouTube answers **HTTP 403 on every download**,
    # audio and video alike. Measured 2026-08-30 on yt-dlp 2026.08.19: android_vr
    # 403s, while the default client set returns the full ladder (1080p60 avc1 +
    # m4a, up to 2160p) and the app's own format string downloads from it.
    # Override via env e.g. YTDLP_PLAYER_CLIENT="android_vr,web_safari" if YouTube
    # flips this back again.
    YTDLP_PLAYER_CLIENT = [
        c.strip()
        for c in os.getenv("YTDLP_PLAYER_CLIENT", "default").split(",")
        if c.strip()
    ]
    YTDLP_EXTRACTOR_ARGS = {"youtube": {"player_client": YTDLP_PLAYER_CLIENT}}

    # yt-dlp unified options - Phase A optimized settings
    YTDLP_SOCKET_TIMEOUT = int(
        os.getenv("YTDLP_SOCKET_TIMEOUT", 30)
    )  # Reduced for faster failure
    YTDLP_RETRIES = int(os.getenv("YTDLP_RETRIES", 5))  # Reduced from 10 to 5
    YTDLP_FRAGMENT_RETRIES = int(
        os.getenv("YTDLP_FRAGMENT_RETRIES", 5)
    )  # Reduced from 10 to 5

    # Phase A: Optimized format for remux-only merge (no re-encoding)
    YTDLP_OPTIMIZED_FORMAT = os.getenv(
        "YTDLP_OPTIMIZED_FORMAT",
        # YouTube-optimized remux path first (avc1+mp4a -> no re-encode)...
        "bestvideo[height<=1080][vcodec^=avc1]+bestaudio[acodec^=mp4a]/"
        "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/"
        # ...then generic adaptive (separate video+audio, e.g. Fox News / TED
        # HLS with non-mp4a/m4a codecs). These come BEFORE the single-file
        # fallbacks so we prefer working HLS streams over progressive URLs that
        # some sites (e.g. TED's h264-1200k) serve but then reject with HTTP 403.
        "bestvideo[height<=1080]+bestaudio/" "bestvideo+bestaudio/"
        # ...single muxed file as last resort (sources with no separate streams).
        "best[ext=mp4]/best",
    )
    YTDLP_CACHE_DIR = os.getenv("YTDLP_CACHE_DIR", "/tmp/yt-dlp")
    YTDLP_RESTRICT_FILENAMES = (
        os.getenv("YTDLP_RESTRICT_FILENAMES", "True").lower() == "true"
    )
    YTDLP_CONTINUE_DL = os.getenv("YTDLP_CONTINUE_DL", "True").lower() == "true"
    YTDLP_MERGE_OUTPUT_FORMAT = os.getenv("YTDLP_MERGE_OUTPUT_FORMAT", "mp4")
    YTDLP_FORMAT_SORT = ["res:1080", "fps", "codec:avc1:m4a", "ext:mp4"]

    # FFmpeg timeouts
    FFPROBE_TIMEOUT = int(os.getenv("FFPROBE_TIMEOUT", 30))
    FFMPEG_RUN_TIMEOUT = int(os.getenv("FFMPEG_RUN_TIMEOUT", 900))

    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv(
        "LOG_FORMAT", "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    )

    # Clean Logging Settings
    ENABLE_JSON_LOGGING = os.getenv("ENABLE_JSON_LOGGING", "False").lower() == "true"
    ENABLE_PERFORMANCE_LOGGING = (
        os.getenv("ENABLE_PERFORMANCE_LOGGING", "True").lower() == "true"
    )
    LOG_EXTERNAL_SERVICES = (
        os.getenv("LOG_EXTERNAL_SERVICES", "False").lower() == "true"
    )  # Reduce noise

    # Progress Update Intervals (seconds)
    DOWNLOAD_PROGRESS_INTERVAL = float(os.getenv("DOWNLOAD_PROGRESS_INTERVAL", 2.0))
    SYSTEM_RESOURCE_CHECK_INTERVAL = float(
        os.getenv("SYSTEM_RESOURCE_CHECK_INTERVAL", 30.0)
    )

    # Progress Logs Exposure (optional)
    EXPOSE_PROGRESS_LOGS = os.getenv("EXPOSE_PROGRESS_LOGS", "False").lower() == "true"
    PROGRESS_LOGS_TAIL = int(os.getenv("PROGRESS_LOGS_TAIL", 50))

    # Download security (optional)
    REQUIRE_DOWNLOAD_TOKEN = (
        os.getenv("REQUIRE_DOWNLOAD_TOKEN", "False").lower() == "true"
    )

    # URL validation security
    ALLOW_UNKNOWN_DOMAINS = (
        os.getenv("ALLOW_UNKNOWN_DOMAINS", "False").lower() == "true"
    )

    # YouTube Download Feature Toggle
    ENABLE_YOUTUBE_DOWNLOAD = (
        os.getenv("ENABLE_YOUTUBE_DOWNLOAD", "True").lower() == "true"
    )

    # Research Recorder — full per-run archive for corpus analysis.
    # Every run writes an immutable, timestamped directory under RESEARCH_DIR: the
    # inputs, every intermediate cue list, every LLM request/response and copies of the
    # outputs. This exists because /app/downloads is keyed by FILENAME, so a second run
    # of the same video overwrites the first and the comparison is lost forever.
    # RESEARCH_DIR lives on the `storage` volume, which survives container rebuilds.
    RESEARCH_DIR = os.getenv("RESEARCH_DIR", "/app/storage/research")
    RESEARCH_RECORDER_ENABLED = (
        os.getenv("RESEARCH_RECORDER_ENABLED", "True").lower() == "true"
    )
    #: Largest single output file the recorder will copy, in MB. The final MP4 is
    #: archived on purpose (owner-approved: the rendered frames ARE the evidence), but a
    #: pathological multi-GB input must not fill the volume silently — over this it is
    #: skipped with a WARNING and recorded as skipped in meta.json.
    RESEARCH_MAX_COPY_MB = int(os.getenv("RESEARCH_MAX_COPY_MB", 4096))

    # Testing/CI Fake Mode Configuration
    USE_FAKE_YTDLP = os.getenv("USE_FAKE_YTDLP", "False").lower() == "true"
    FAKE_ASSETS_DIR = os.getenv("FAKE_ASSETS_DIR", "/app/test_assets")
    FAKE_VIDEO_SOURCE = os.getenv("FAKE_VIDEO_SOURCE", "test_video.mp4")

    @classmethod
    def get_watermark_path(cls, watermark_type: str = "default") -> str:
        """Get watermark path by type"""
        return cls.WATERMARK_PATHS.get(watermark_type, cls.WATERMARK_PATHS["default"])

    @classmethod
    def is_allowed_file_extension(cls, filename: str) -> bool:
        """Check if file extension is allowed"""
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in cls.ALLOWED_EXTENSIONS
        )

    @classmethod
    def is_valid_whisper_model(cls, model: str) -> bool:
        """Check if whisper model is valid"""
        return model in cls.AVAILABLE_WHISPER_MODELS

    @classmethod
    def is_model_restricted(cls, model: str) -> bool:
        """
        Check if a whisper model is restricted in hosted mode.
        Returns True if: HOSTED_MODE is True AND model is in PRO_ONLY_MODELS
        Returns False for self-hosted deployments (all models allowed)
        """
        return cls.HOSTED_MODE and model in cls.PRO_ONLY_MODELS

    @classmethod
    def is_youtube_restricted(cls) -> bool:
        """
        Check if YouTube/Online Video feature is restricted.
        Returns True if: HOSTED_MODE is True (YouTube is PRO-only in hosted mode)
        Returns False for self-hosted deployments (YouTube is available)
        """
        return cls.HOSTED_MODE

    @classmethod
    def get_supported_language_name(cls, lang_code: str) -> str:
        """Get language display name by code"""
        return cls.SUPPORTED_LANGUAGES.get(lang_code, lang_code)


class DevelopmentConfig(Config):
    """Development configuration.

    Every value here is an env-overridable *default*, never a hard override.
    Plain class attributes (``DEBUG = True``) silently beat anything the
    operator puts in the environment or in docker-compose, which is how
    ``DEBUG=False`` + ``LOG_LEVEL=INFO`` in docker-compose ended up being
    ignored. Read the env first and fall back to the development-friendly
    value so the environment always wins.
    """

    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
    WORKER_CONCURRENCY = int(
        os.getenv("WORKER_CONCURRENCY", 1)
    )  # Lower for development
    ALLOW_UNKNOWN_DOMAINS = (
        os.getenv("ALLOW_UNKNOWN_DOMAINS", "True").lower() == "true"
    )  # Allow unknown domains in development

    # Use local directories for development
    _BASE = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", os.path.join(_BASE, "uploads"))
    DOWNLOADS_FOLDER = os.getenv("DOWNLOADS_FOLDER", os.path.join(_BASE, "downloads"))
    WHISPER_MODELS_FOLDER = os.getenv(
        "WHISPER_MODELS_FOLDER", os.path.join(_BASE, "whisper_models")
    )
    ASSETS_FOLDER = os.getenv("ASSETS_FOLDER", os.path.join(_BASE, "assets"))

    # Update watermark paths for development
    WATERMARK_PATHS: dict[str, str] = {
        "default": os.path.join(ASSETS_FOLDER, "logo.png"),
        "alternative": os.path.join(ASSETS_FOLDER, "watermark.png"),
    }


class ProductionConfig(Config):
    """Production configuration"""

    # Intentionally NOT env-overridable: debug mode leaks stack traces and the
    # interactive debugger, so production pins it off regardless of the env.
    DEBUG = False
    # Operational knobs: env wins, these are only defaults.
    LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")
    WORKER_CONCURRENCY = int(
        os.getenv("WORKER_CONCURRENCY", 4)
    )  # Higher for production


class TestingConfig(Config):
    """Testing configuration"""

    DEBUG = True
    TESTING = True
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB for testing
    # Use local backend folders by default to avoid read-only '/app' paths
    _BASE = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", os.path.join(_BASE, "uploads"))
    DOWNLOADS_FOLDER = os.getenv("DOWNLOADS_FOLDER", os.path.join(_BASE, "downloads"))
    CELERY_TASK_ALWAYS_EAGER = True  # Execute tasks synchronously
    CELERY_TASK_EAGER_PROPAGATES = True
    CELERY_TASK_STORE_EAGER_RESULT = True  # Store results so AsyncResult works
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "memory://")
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "cache+memory://")


# Configuration factory
def get_config() -> Config:
    """Get configuration based on environment"""
    env = os.getenv("FLASK_ENV", "production").lower()

    if env == "development":
        return DevelopmentConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return ProductionConfig()
