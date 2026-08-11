#!/usr/bin/env python3
"""
Smart Whisper Model Selection with FASTER-WHISPER
Automatically chooses the best model based on language and content
MUCH FASTER THAN REGULAR WHISPER! 🚀

Now supports Gemini API for YouTube transcription!
"""

import logging
import os

import numpy as np
from faster_whisper import WhisperModel

# Import Gemini transcription if available
try:
    from services.gemini_transcription import transcribe_with_gemini

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    transcribe_with_gemini = None

logger = logging.getLogger(__name__)

#: Free memory a ``large`` load needs before it is safe to attempt. Unchanged from the
#: guard that has always been here — only the NUMBER it was comparing against was wrong.
LARGE_MODEL_MIN_FREE_GB = 6.0

#: Below this, faster-whisper's beam search is what tips a container over, so the search
#: is narrowed rather than the model swapped.
LOW_MEMORY_BEAM_1_GB = 2.0

#: cgroup v1 spells "no limit" as a near-2**63 sentinel (9223372036854771712). Anything
#: above 8 PiB is that sentinel, not a machine.
_CGROUP_NO_LIMIT_BYTES = 1 << 53

_GIB = 1024**3


def _log_downgrade(requested: str, actual: str, reason: str) -> str:
    """Log a model downgrade at WARNING and return the model that will actually run.

    Every downgrade in this module used to be invisible — some were logged at INFO,
    some not at all — while the caller went on recording the model it ASKED for. That
    is how Redis ended up with 104 rows indexed ``stats:index:model:large`` whose runs
    may have executed ``medium``. A downgrade is a change to the product the user
    receives, so it is a WARNING, it names requested -> actual, and it says why.
    """
    logger.warning(
        f"⚠️ MODEL DOWNGRADE: requested '{requested}' -> running '{actual}' ({reason})"
    )
    return actual


def resolve_model(requested: str | None) -> str:
    """The model that will actually be loaded for ``requested``, memory taken into account.

    This decision existed only INSIDE :meth:`SmartWhisperManager.transcribe_smart` — and
    both of the pipeline's real entry points (``transcribe_and_translate_streamed`` and
    ``transcribe_with_words``) call :meth:`load_model` directly and never go through it.
    So the memory guard, the ``small`` mapping and every downgrade WARNING were dead code
    on the two paths that matter; the only place they ran was a helper nothing calls in
    production. One function, called from all three, so the guard cannot be bypassed by
    adding a fourth caller.

    Returns the requested name unchanged when it is already safe, and a downgraded name
    (logged at WARNING by :func:`_log_downgrade`) when it is not.
    """
    if not requested:
        return "tiny"
    if requested == "small":
        return _log_downgrade(
            "small",
            "medium",
            "this service does not load 'small'; 'medium' is its nearest model",
        )
    if requested != "large":
        return requested

    headroom_gb, source = memory_headroom_gb()
    if headroom_gb is None:
        # UNKNOWN is not LOW. Downgrading here is what made every `large` request on a
        # host without cgroup or /proc/meminfo (i.e. every native macOS dev run) produce
        # `medium` output while the UI, the stats and the archive all said `large`.
        logger.warning(
            f"⚠️ Cannot measure free memory ({source}) — honouring the requested 'large' "
            f"model with UNKNOWN headroom. If this process is killed, start here."
        )
        return "large"
    if headroom_gb < LARGE_MODEL_MIN_FREE_GB:
        return _log_downgrade(
            "large",
            "medium",
            f"only {headroom_gb:.2f} GiB free ({source}), below the "
            f"{LARGE_MODEL_MIN_FREE_GB:.1f} GiB 'large' needs",
        )
    logger.info(f"✅ {headroom_gb:.2f} GiB free ({source}), proceeding with 'large'")
    return "large"


def _read_int_file(path: str) -> int | None:
    """Return the single integer in ``path``, or None if it is missing or not a number.

    cgroup v2 writes the literal string ``max`` into ``memory.max`` for an unlimited
    cgroup, which is an answer ("no limit here"), not an error — both come back as None
    and the caller moves on to the next source.
    """
    try:
        with open(path) as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def _cgroup_memory(limit_path: str, current_path: str, stat_path: str):
    """``(limit_gb, working_set_gb)`` for one cgroup layout, or None if it is not there."""
    limit = _read_int_file(limit_path)
    current = _read_int_file(current_path)
    if limit is None or current is None or limit >= _CGROUP_NO_LIMIT_BYTES:
        return None

    # Page cache counts towards `current` but the kernel hands it back under pressure,
    # so the non-reclaimable "working set" is current minus the inactive file cache —
    # the same subtraction the kernel's own OOM accounting (and Kubernetes) makes.
    # Measured in this project's idle worker: current 0.39 GiB, of which inactive_file
    # is 0.17 GiB. Counting that cache as "in use" would understate free memory by
    # nearly half and downgrade `large` for no reason.
    inactive_file = 0
    try:
        with open(stat_path) as handle:
            for line in handle:
                key, _, value = line.partition(" ")
                if key in ("inactive_file", "total_inactive_file"):
                    inactive_file = int(value.strip())
                    break
    except (OSError, ValueError):
        inactive_file = 0

    working_set = max(current - inactive_file, 0)
    return limit / _GIB, working_set / _GIB


def memory_headroom_gb() -> tuple[float | None, str]:
    """How much memory this process can still take, and where the number came from.

    Returns ``(None, reason)`` when nothing readable can answer. ``None`` means UNKNOWN
    and callers must not read it as "low" — see below.

    Why this is not simply ``/proc/meminfo``, measured inside this project's own worker
    container: the cgroup limit is 8.00 GiB against 2.98 GiB in use, i.e. 5.02 GiB of
    real headroom — under the 6.0 GiB bar ``large`` needs — while ``/proc/meminfo``
    reported 14.97 GiB ``MemAvailable``, because ``/proc`` describes the HOST VM and not
    the container the model has to fit inside. The guard read the host's number and
    waved ``large`` through every time.

    And on native macOS ``/proc/meminfo`` does not exist at all, so the old probe raised
    ``FileNotFoundError`` on every single call and its ``except`` branch silently turned
    every ``large`` request into ``medium`` — verified on this machine.
    """
    cgroup = _cgroup_memory(  # cgroup v2 (Docker Desktop, modern Linux)
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory.stat",
    ) or _cgroup_memory(  # cgroup v1
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        "/sys/fs/cgroup/memory/memory.stat",
    )
    if cgroup:
        limit_gb, used_gb = cgroup
        return (
            limit_gb - used_gb,
            f"cgroup limit {limit_gb:.2f} GiB, {used_gb:.2f} GiB in use",
        )

    # No cgroup limit: the host's own free memory IS the bound that applies.
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return (
                        int(line.split()[1]) / (1024**2),
                        "/proc/meminfo MemAvailable (host-wide, no cgroup limit)",
                    )
    except (OSError, ValueError, IndexError):
        pass

    return None, "no cgroup memory limit and no readable /proc/meminfo"


class SmartWhisperManager:
    def __init__(self):
        self.loaded_models: dict[str, WhisperModel] = {}

        # The model that ACTUALLY ran, last time anything ran here. Callers only know
        # what they REQUESTED, and the two diverge on every downgrade path below; this
        # is the only place that knows which one really executed, so it is published
        # for the stats writer to record instead of the request.
        # Safe as instance state: the worker runs --concurrency=1 --max-tasks-per-child=1
        # (docker-compose.yml), i.e. one job per process, and every load happens on the
        # same thread that then transcribes.
        self.last_model_used: str | None = None

        # Set up persistent cache directory for models
        # Default to local directory in dev, Docker will override with env var
        default_cache = os.path.join(os.getcwd(), "whisper_models")
        self.cache_dir = os.getenv("WHISPER_MODELS_DIR", default_cache)

        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except (OSError, PermissionError) as e:
            logger.warning(
                f"Could not create cache dir {self.cache_dir}: {e}. Using temp dir."
            )
            import tempfile

            self.cache_dir = tempfile.mkdtemp(prefix="whisper_cache_")
        # Support for LARGE and MEDIUM models with faster-whisper 🚀
        # Plus GEMINI API for experimental YouTube transcription
        self.model_capabilities = {
            "base": {
                "languages": "all",
                "max_duration": float("inf"),
                "accuracy": "good",  # Upgraded from "fair" - base is solid
                "speed": "fast",
            },
            "medium": {
                "languages": "all",  # Supports all languages
                "max_duration": float("inf"),
                "accuracy": "very_good",
                "speed": "balanced",  # More accurate description
            },
            "large": {
                "languages": "all",  # Supports all languages
                "max_duration": float("inf"),
                "accuracy": "excellent",
                "speed": "thorough",
            },
            "gemini": {
                "languages": "all",
                "max_duration": 900,  # 15 minutes
                "accuracy": "experimental",
                "speed": "very_fast",
                "requires_youtube": True,  # Only works with YouTube URLs
                "requires_api_key": True,
            },
        }

        # VIP languages that benefit from larger models
        self.vip_languages = {
            "he": "medium",  # Hebrew works better with medium+
            "ar": "medium",  # Arabic works better with medium+
            "zh": "small",  # Chinese needs at least small
            "ja": "small",  # Japanese needs at least small
            "ko": "small",  # Korean needs at least small
            "ru": "small",  # Russian needs at least small
            "th": "medium",  # Thai needs medium+
            "hi": "medium",  # Hindi needs medium+
        }

    def choose_model(
        self,
        language: str = "auto",
        duration: float | None = None,
        quality_preference: str = "balanced",
        model_preference: str = None,
    ) -> str:
        """
        Choose between Large and Medium models
        """
        if model_preference and model_preference in self.model_capabilities:
            if model_preference == "large":
                logger.info("🏆 Using LARGE model for maximum accuracy!")
            elif model_preference == "medium":
                logger.info("⚡ Using MEDIUM model for faster processing!")
            elif model_preference == "tiny":
                logger.info("💨 Using TINY model for maximum speed!")
            return model_preference

        # Default to tiny if no preference
        logger.info("💨 Using TINY model for maximum speed (default)!")
        return "tiny"

    def load_model(self, model_name: str) -> WhisperModel:
        """Load and cache FASTER-WHISPER model"""
        if model_name not in self.loaded_models:
            logger.info(
                f"🚀 === LOADING FASTER-WHISPER MODEL: {model_name.upper()} ==="
            )
            try:
                logger.info(f"🌐 Loading {model_name} model with faster-whisper...")
                logger.info("⚡ This is MUCH faster than regular whisper!")

                # Use CPU-optimized compute type - int8 is faster and more compatible
                if model_name in ["large", "medium"]:
                    compute_type = "int8"  # CPU-optimized, 2x faster than float32
                    logger.info(f"💾 Using CPU-optimized compute_type: {compute_type}")
                else:
                    compute_type = "int8"  # Consistent optimization for all models

                # Use faster-whisper with memory optimizations
                self.loaded_models[model_name] = WhisperModel(
                    model_name,
                    device="cpu",
                    compute_type=compute_type,
                    download_root=self.cache_dir,
                )
                logger.info(
                    f"✅ Model {model_name} loaded with faster-whisper successfully"
                )
            except Exception as e:
                logger.error(f"Failed to load {model_name} model: {e}")
                # Fallback to base model
                if model_name != "base":
                    _log_downgrade(model_name, "base", f"it failed to load: {e}")
                    return self.load_model("base")
                raise

        # Only now is the claim true: this model is loaded and is the one that will run.
        self.last_model_used = model_name
        return self.loaded_models[model_name]

    def preload_large_model(self):
        """Download only the LARGE model for maximum accuracy with faster-whisper"""
        model_name = "large"

        try:
            logger.info(
                f"🏆 Downloading {model_name.upper()} model for maximum accuracy..."
            )
            logger.info("⚡ Using faster-whisper for much better performance!")
            # This will download the model if not cached
            self.load_model(model_name)
            logger.info(f"✅ Successfully downloaded {model_name.upper()} model!")
        except Exception as e:
            logger.error(f"❌ Failed to download {model_name}: {e}")

    def cleanup_old_models(self):
        """Remove all models except LARGE"""
        models_to_remove = ["tiny", "base", "small", "medium"]
        removed_count = 0

        for model_name in models_to_remove:
            model_path = os.path.join(self.cache_dir, f"{model_name}.pt")
            if os.path.exists(model_path):
                try:
                    os.remove(model_path)
                    logger.info(f"🗑️ Removed {model_name} model")
                    removed_count += 1
                except Exception as e:
                    logger.error(f"❌ Failed to remove {model_name}: {e}")

        if removed_count > 0:
            logger.info(f"🧹 Cleanup complete! Removed {removed_count} old models")
        else:
            logger.info("✨ Cache already clean - only LARGE model present!")

    def get_cached_models(self):
        """Get list of models that are cached locally"""
        cached = []
        models_to_check = ["tiny", "base", "small", "medium", "large"]

        for model_name in models_to_check:
            model_path = os.path.join(self.cache_dir, f"{model_name}.pt")
            if os.path.exists(model_path):
                cached.append(model_name)

        return cached

    def transcribe_smart(
        self,
        audio_input: str | np.ndarray | None,
        language: str = "auto",
        duration: float | None = None,
        quality_preference: str = "balanced",
        model_preference: str = None,
        model_callback=None,
        progress_callback=None,
        youtube_url: str | None = None,  # NEW: For Gemini support
    ):
        """
        Smart transcription with optimal model selection

        Args:
            audio_input: Path to audio file OR NumPy array of audio data OR None (for Gemini)
            language: Language code or 'auto' for detection
            duration: Audio duration in seconds (for optimization)
            quality_preference: 'speed', 'balanced', or 'quality'
            model_preference: Force specific model (overrides smart selection)
            youtube_url: YouTube URL (required for Gemini)

        Returns:
            Transcription result
        """
        logger.info("🔍 === TRANSCRIBE_SMART CALLED ===")
        logger.info(f"   model_preference={model_preference}")
        logger.info(f"   youtube_url={youtube_url}")
        logger.info(f"   language={language}")
        logger.info("🎯 === SMART MODEL SELECTION ===")

        # === GEMINI PATH ===
        if model_preference == "gemini":
            if not youtube_url:
                model_preference = _log_downgrade(
                    "gemini", "base", "Gemini needs a YouTube URL and none was given"
                )
            else:
                logger.info("🚀 Using Gemini API for YouTube transcription")
                try:
                    if model_callback:
                        model_callback()

                    result = transcribe_with_gemini(
                        youtube_url=youtube_url,
                        language=language,
                        progress_callback=progress_callback,
                    )

                    logger.info("✅ Gemini transcription successful!")
                    # Gemini never touches load_model, so record here what ran — and
                    # hand it back in the payload as well, so a caller that keeps the
                    # dict does not have to ask this manager afterwards.
                    self.last_model_used = "gemini"
                    if isinstance(result, dict):
                        result.setdefault("model_used", "gemini")
                    return result

                except Exception as e:
                    logger.error(f"❌ Gemini failed: {e}")
                    model_preference = _log_downgrade(
                        "gemini", "base", f"Gemini transcription failed: {e}"
                    )
                    # Continue to Whisper below

        # === WHISPER PATH (original code) ===
        if isinstance(audio_input, str):
            logger.info(f"🎵 Audio: {os.path.basename(audio_input)}")
            if duration is None:
                # Try to get duration from file if not provided
                try:
                    import ffmpeg

                    duration = float(ffmpeg.probe(audio_input)["format"]["duration"])
                except Exception:
                    pass  # Duration will remain unknown
        else:
            logger.info("🎵 Audio: from in-memory buffer")

        logger.info(f"🌍 Language: {language}")
        logger.info(
            f"⏱️ Duration: {duration:.2f}s" if duration else "⏱️ Duration: unknown"
        )
        logger.info(f"🎛️ Quality preference: {quality_preference}")
        logger.info(f"🔧 Model preference: {model_preference or 'auto'}")

        # Choose optimal model
        if model_preference:
            logger.info(f"🎯 Using forced model: {model_preference}")
            # Support tiny, base, medium and large, map others to large
            if model_preference in ["tiny", "base", "medium", "large", "small"]:
                model_name = resolve_model(model_preference)
            else:
                model_name = model_preference
        else:
            model_name = self.choose_model(
                language, duration, quality_preference, model_preference
            )
            logger.info(f"🎯 Smart model selected: {model_name}")

        model = self.load_model(model_name)

        # Signal that model is loaded
        if model_callback:
            model_callback()

        # Memory-optimized transcription options
        options = {
            "word_timestamps": True,  # True is better check i it
            "beam_size": (
                2 if model_name in ["large", "medium"] else 5
            ),  # Reduce beam_size for large models
            "chunk_length": 30,  # Process in 30-second chunks for better performance (10-15% speedup)
            "condition_on_previous_text": True,  # Reduces memory usage between chunks
        }

        # Add language hint if specified
        if language != "auto":
            options["language"] = language

        logger.info(
            f"💾 Memory-optimized settings: beam_size={options['beam_size']}, chunk_length={options['chunk_length']}s"
        )

        logger.info(f"Starting transcription with {model_name} model")

        # Monitor memory before transcription. Same probe as the model guard above, and
        # for the same reason: this used to read the HOST's free memory from
        # /proc/meminfo while the decode has to fit inside the container's cgroup.
        headroom_gb, source = memory_headroom_gb()
        if headroom_gb is None:
            logger.warning(
                f"💾 Memory headroom unknown ({source}) — keeping "
                f"beam_size={options['beam_size']}"
            )
        else:
            logger.info(
                f"💾 Memory headroom before transcription: {headroom_gb:.2f} GiB "
                f"({source})"
            )
            if headroom_gb < LOW_MEMORY_BEAM_1_GB:
                logger.warning(
                    f"⚠️ Low memory ({headroom_gb:.2f} GiB free) - reducing "
                    f"beam_size to 1"
                )
                options["beam_size"] = 1

        try:
            # Calculate expected chunks for progress tracking
            audio_duration = duration
            if not audio_duration and isinstance(audio_input, np.ndarray):
                audio_duration = len(audio_input) / 16000  # 16kHz sample rate

            chunk_length = options.get("chunk_length", 20)
            expected_chunks = (
                int(audio_duration / chunk_length) if audio_duration else None
            )

            if progress_callback and expected_chunks:
                logger.info(
                    f"📊 Transcription progress: Expected ~{expected_chunks} chunks of {chunk_length}s each"
                )
                logger.info(
                    f"🎙️ Starting transcription: 0s/{audio_duration:.0f}s (0.0%)"
                )

            # Create a progress-aware transcription wrapper
            if progress_callback and expected_chunks:
                segments_processed = 0
                all_segments = []

                # Process with progress tracking
                segments_iter, info = model.transcribe(audio_input, **options)

                for segment in segments_iter:
                    all_segments.append(segment)
                    segments_processed += 1

                    # Estimate progress based on segment timing
                    if hasattr(segment, "end") and audio_duration:
                        # Calculate progress: 30% to 85% (55% total range for transcription)
                        transcription_progress = (
                            segment.end / audio_duration
                        ) * 100  # 0-100%
                        step_progress = 30 + int(
                            transcription_progress * 0.55
                        )  # 30-85%
                        overall_progress = (
                            step_progress  # Same as step progress for this phase
                        )

                        progress_callback(
                            step_progress,
                            f"Transcription: {segment.end:.0f}s/{audio_duration:.0f}s",
                            overall_progress,
                            "Step 1: Whisper AI",
                            5,
                        )

                        # Log progress every ~20 seconds or at milestones
                        if segments_processed % 10 == 0 or transcription_progress >= 95:
                            logger.info(
                                f"🎙️ Transcription progress: {segment.end:.0f}s/{audio_duration:.0f}s ({transcription_progress:.1f}%)"
                            )

                segments = all_segments
            else:
                # Standard transcription without progress
                segments, info = model.transcribe(audio_input, **options)

            # Convert to OpenAI Whisper format for compatibility
            result = {
                "text": "",
                "segments": [],
                "language": info.language,
                "model_used": model_name,
                "model_info": self.model_capabilities.get(
                    model_name, self.model_capabilities["large"]
                ),
            }

            # Process segments
            for segment in segments:
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
                result["segments"].append(segment_dict)
                result["text"] += segment.text

            logger.info(
                f"Transcription completed with {model_name} model. Detected language: {result.get('language', 'unknown')}"
            )
            return result

        except Exception as e:
            logger.error(f"Transcription failed with {model_name} model: {e}")

            # Try fallback to base model if not already using it
            if model_name != "base":
                _log_downgrade(model_name, "base", f"transcription raised: {e}")
                return self.transcribe_smart(
                    audio_input, language, duration, "speed", model_preference="base"
                )

            raise

    def get_model_info(self, model_name: str) -> dict:
        """Get information about a model"""
        return self.model_capabilities.get(model_name, {})

    def get_available_models(self) -> dict:
        """Get all available models and their capabilities"""
        return self.model_capabilities

    def unload_model(self, model_name: str):
        """Unload a model to free memory"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            logger.info(f"Unloaded {model_name} model")

    def unload_all_models(self):
        """Unload all models to free memory"""
        self.loaded_models.clear()
        logger.info("Unloaded all models")

    def get_memory_usage(self) -> dict[str, str]:
        """Get approximate memory usage of loaded models"""
        model_sizes = {
            "tiny": "39 MB",
            "base": "74 MB",
            "small": "244 MB",
            "medium": "769 MB",
            "large": "1550 MB",
        }

        usage = {}
        total_mb = 0

        for model_name in self.loaded_models:
            size_str = model_sizes.get(model_name, "Unknown")
            usage[model_name] = size_str
            if "MB" in size_str:
                total_mb += int(size_str.split(" ")[0])

        usage["total"] = f"{total_mb} MB"
        return usage


# Global smart whisper manager
smart_whisper = SmartWhisperManager()


def detect_audio_language(audio_path: str) -> tuple[str, float]:
    """
    Detect language of audio file using faster-whisper

    Returns:
        (language_code, confidence)
    """
    try:
        # Use large model (it's fast enough with faster-whisper)
        model = smart_whisper.load_model("large")

        # Transcribe just the first 30 seconds for language detection
        segments, info = model.transcribe(
            audio_path,
            language=None,
            word_timestamps=False,
            beam_size=1,  # Faster detection
        )

        language = info.language
        # Use language probability as confidence
        confidence = info.language_probability

        logger.info(f"Detected language: {language} (confidence: {confidence:.2f})")
        return language, confidence

    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return "unknown", 0.0
