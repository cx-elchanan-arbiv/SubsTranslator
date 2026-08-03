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


class SmartWhisperManager:
    def __init__(self):
        self.loaded_models: dict[str, WhisperModel] = {}

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
                    logger.info("Falling back to base model")
                    return self.load_model("base")
                raise

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
                logger.warning("⚠️ Gemini requires YouTube URL, falling back to base")
                model_preference = "base"
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
                    return result

                except Exception as e:
                    logger.error(f"❌ Gemini failed: {e}")
                    logger.info("🔄 Falling back to Whisper (base)")
                    model_preference = "base"
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
            if model_preference in ["tiny", "base", "medium", "large"]:
                # Check memory before using large model
                if model_preference == "large":
                    try:
                        # Check available memory using /proc/meminfo
                        with open("/proc/meminfo") as f:
                            meminfo = f.read()
                        available_kb = int(
                            [
                                line
                                for line in meminfo.split("\n")
                                if "MemAvailable" in line
                            ][0].split()[1]
                        )
                        available_gb = available_kb / (1024**2)

                        if available_gb < 6.0:  # Less than 6GB available
                            logger.warning(
                                f"⚠️ Only {available_gb:.1f}GB RAM available, using 'medium' instead of 'large'"
                            )
                            model_name = "medium"
                        else:
                            logger.info(
                                f"✅ {available_gb:.1f}GB RAM available, proceeding with 'large' model"
                            )
                            model_name = model_preference
                    except (FileNotFoundError, IndexError, ValueError):
                        logger.warning(
                            "⚠️ Cannot check memory, using 'medium' instead of 'large' for safety"
                        )
                        model_name = "medium"
                else:
                    model_name = model_preference
            elif model_preference in ["small"]:
                logger.info(
                    f"🔄 Mapping {model_preference} to 'medium' for better compatibility"
                )
                model_name = "medium"
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

        # Monitor memory before transcription
        try:
            with open("/proc/meminfo") as f:
                meminfo = f.read()
            available_kb = int(
                [line for line in meminfo.split("\n") if "MemAvailable" in line][
                    0
                ].split()[1]
            )
            available_gb = available_kb / (1024**2)
            logger.info(
                f"💾 Memory available before transcription: {available_gb:.1f}GB"
            )

            if available_gb < 2.0:
                logger.warning(
                    f"⚠️ Low memory ({available_gb:.1f}GB) - reducing beam_size to 1"
                )
                options["beam_size"] = 1
        except:
            pass

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
                logger.info("Trying fallback to base model")
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
