# SubsTranslator - Dependency & Module Map

## Module Dependency Graph

```
┌─────────────────────────────────────────────────────┐
│  ENTRY POINTS                                        │
├─────────────────────────────────────────────────────┤
│  • app.py (Flask routes - Port 8081)                │
│  • celery_worker.py (Background tasks)              │
│  • celery_config.py (Celery configuration)          │
└────────────┬────────────────────────────────────────┘
             │
             ├────────────────────────────────────────────┐
             │                                            │
             ▼                                            ▼
    ┌────────────────────┐                      ┌─────────────────┐
    │ TASK ORCHESTRATION │                      │  REQUEST LAYER  │
    ├────────────────────┤                      ├─────────────────┤
    │ tasks.py (2,535 L) │───────┬──────────┬──│ app.py (2,231 L)│
    │                    │       │          │  │                 │
    │ Task Functions:    │       │          │  │ Routes:         │
    │ • download_&_process│      │          │  │ • /upload       │
    │ • process_video    │       │          │  │ • /youtube      │
    │ • translate_*      │       │          │  │ • /status       │
    │ • embed_subtitles  │       │          │  │ • /api/stats/*  │
    │ • merge_videos     │       │          │  │ • /health       │
    └─┬─────────────────┘       │          │  └────────┬────────┘
      │                         │          │           │
      ├─────────────────────────┼──────────┼───────────┤
      │                         │          │           │
      ▼                         ▼          ▼           ▼
┌──────────────────────────────────────────────────────────┐
│         CORE PROCESSING MODULES                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  TRANSCRIPTION:                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ whisper_smart.py (522 L)                        │   │
│  │ • SmartWhisperManager                           │   │
│  │ • Model selection (tiny/base/medium/large)      │   │
│  │ • imports: faster_whisper, numpy                │   │
│  │ • optionally imports: gemini_transcription.py   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
│  GEMINI TRANSCRIPTION (NEW):                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │ gemini_transcription.py (346 L)                 │   │
│  │ • GeminiTranscriptionError                      │   │
│  │ • parse_timestamp()                             │   │
│  │ • transcribe_with_gemini()                      │   │
│  │ • imports: google.genai, yt_dlp                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
│  TRANSLATION:                                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │ translation_services.py (646 L)                 │   │
│  │ • GoogleTranslator                              │   │
│  │ • OpenAITranslator                              │   │
│  │ • get_translator()                              │   │
│  │ • imports: deep_translator, openai, tiktoken    │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
│  VIDEO PROCESSING:                                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │ video_utils.py (516 L)                          │   │
│  │ • cut_video_ffmpeg()                            │   │
│  │ • embed_subtitles_ffmpeg()                      │   │
│  │ • parse_text_to_srt()                           │   │
│  │ • add_watermark_to_video()                      │   │
│  │ • imports: ffmpeg-python, subprocess            │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
└──────────────────────────────────────────────────────────┘
      │                  │                 │
      │                  │                 │
      └──────┬───────────┴─────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────┐
│         SUPPORT & SERVICE MODULES                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  CONFIGURATION & MANAGEMENT:                            │
│  • config.py (297 L) - Settings & defaults              │
│  • state_manager.py (296 L) - Processing state tracking │
│  • shared_config.py (204 L) - Shared configuration      │
│  • metadata_service.py (260 L) - Video metadata         │
│                                                          │
│  UTILITIES & HELPERS:                                   │
│  • logging_config.py (226 L) - Structured logging       │
│  • rtl_utils.py (216 L) - Hebrew/RTL text processing    │
│  • file_probe.py (246 L) - Video file analysis          │
│  • ytdlp_hooks.py (250 L) - YouTube-DL progress hooks   │
│  • logo_manager.py (4.8 KB) - Watermark management      │
│  • openai_rate_limiter.py (309 L) - API rate limiting   │
│  • quality_gate.py (5.0 KB) - SRT quality validation    │
│  • phase_logger.py (261 L) - Phase-based logging        │
│  • performance_monitor.py (295 L) - Performance metrics  │
│                                                          │
│  SERVICES:                                              │
│  • services/stats_service.py - Statistics tracking      │
│  • services/subtitle_service.py - Subtitle operations   │
│  • i18n/translations.py - Internationalization          │
│                                                          │
└──────────────────────────────────────────────────────────┘
      │
      │
      └──────────────────────────────────┬─────────────────┐
                                         │                 │
                    ┌────────────────────┴──────┐           │
                    │                           │           │
                    ▼                           ▼           ▼
            ┌──────────────────┐       ┌──────────────┐ ┌─────────────┐
            │   FLASK IMPORTS  │       │  CELERY DEPS │ │ EXTERNAL    │
            ├──────────────────┤       ├──────────────┤ ├─────────────┤
            │ • Flask 3.1.2    │       │ • Celery 5.4 │ │ • Redis 5.2 │
            │ • Flask-CORS 6.0 │       │ • Redis 5.2  │ │ • FFmpeg    │
            │ • Flask-Limiter  │       │ • Kombu 5.4  │ │ • yt-dlp    │
            │ • Werkzeug 3.1.3 │       │ • billiard   │ │ • OpenAI    │
            └──────────────────┘       └──────────────┘ └─────────────┘
```

## Detailed Module Dependencies

### Frontend (React/TypeScript)

```
src/
├── App.tsx (Main entry, 22KB)
│   ├── imports: components/*, hooks/useApi, i18n/*, contexts/AuthContext
│   ├── Uses: Firebase, React Router, Tailwind
│   └── Provides: Main application logic
│
├── components/ (25 files)
│   ├── UploadForm.tsx
│   ├── YoutubeForm.tsx
│   ├── EmbedSubtitlesForm.tsx
│   ├── ProgressDisplay.tsx
│   ├── ResultsDisplay.tsx
│   ├── VideoMergerForm.tsx
│   ├── VideoInfoDisplay.tsx
│   ├── LanguageSelector.tsx
│   ├── WatermarkSettings.tsx
│   ├── AILoader.tsx
│   ├── ErrorCard.tsx
│   ├── Header.tsx
│   ├── HeroSection.tsx
│   ├── Options.tsx
│   ├── Stage.tsx
│   ├── Tabs.tsx
│   ├── AuthModal.tsx
│   ├── LogoOnlyForm.tsx
│   ├── ProtectedRoute.tsx
│   ├── UserProfile.tsx
│   └── [+ 5 more]
│
├── hooks/
│   └── useApi.ts (API communication hook)
│
├── contexts/
│   └── AuthContext.tsx (Auth state management)
│
├── firebase/
│   ├── config.ts
│   └── auth.ts
│
├── i18n/ (Internationalization)
│   ├── config.ts
│   ├── i18n.ts
│   ├── I18nProvider.tsx
│   ├── TranslationContext.tsx
│   └── locales/
│
├── types/ (TypeScript definitions)
│   ├── api.ts
│   ├── errors.ts
│   ├── validation.ts
│   └── index.ts
│
└── utils/
    ├── userPreferences.ts
    ├── apiValidation.ts
    └── __tests__/
```

### Backend Core Dependencies

```
IMMEDIATE DEPENDENCIES (called directly by app.py/tasks.py):
├── tasks.py
├── config.py
├── whisper_smart.py (smart_whisper object)
├── translation_services.py (get_translator)
├── video_utils.py (multiple functions)
├── file_probe.py (probe_file_safe)
├── services/stats_service.py (save_video_stats)
├── services/subtitle_service.py (subtitle_service)
├── logging_config.py (get_logger, log_phase)
├── metadata_service.py (metadata_service)
├── openai_rate_limiter.py (RateLimiter)
├── quality_gate.py (QualityGate)
├── logo_manager.py (LogoManager)
├── phase_logger.py (PhaseLogger)
├── performance_monitor.py (performance_monitor)
├── i18n/translations.py (init_i18n, t)
├── rtl_utils.py (rtl text processing)
├── ytdlp_hooks.py (yt-dlp hooks)
└── state_manager.py (processing state)

TASKS.PY SPECIFIC DEPENDENCIES:
├── whisper_smart.py (smart_whisper)
├── gemini_transcription.py (transcribe_with_gemini)
├── translation_services.py
├── video_utils.py
├── metadata_service.py
├── config.py
├── logging_config.py
├── rtl_utils.py
├── quality_gate.py
├── state_manager.py
└── services/* (stats, subtitles)
```

## File Status & Usage

### ✅ ACTIVELY USED (Modified Nov 20-23):
```
tasks.py .......................... Nov 23 18:22 (Celery tasks)
app.py ........................... Nov 20 13:40 (Flask routes)
gemini_transcription.py ........... Nov 23 17:54 (NEW: Gemini support)
whisper_smart.py ................. Nov 23 17:47 (Smart model selection)
test_gemini_e2e.py ............... Nov 23 18:39 (Gemini testing)
test_gemini_integration.py ........ Nov 23 15:58 (Gemini integration)
state_manager.py ................. Nov 20 11:38 (State management)
metadata_service.py .............. Nov 20 11:36 (Metadata extraction)
```

### 🟡 POTENTIALLY UNUSED / LEGACY:
```
download_video_task.py ........... NOT imported
fix_existing_srt.py .............. Standalone script
fix_srt_file.py .................. Standalone script
retry_missing_segments.py ......... NOT imported
google_translate_with_batches.py . Likely superseded
get_video_metadata.py ............ May be superseded by metadata_service.py
```

### ⚠️ SCATTERED/MISPLACED:
```
/backend/test_*.py (8 files) ..... Should be in /backend/tests/
/test_*.py at project root (5) ... Should be in /backend/tests/ or root tests/
/e2e/helpers_backup/ ............ Should be archived or removed
/e2e/tests_backup/ .............. Should be archived or removed
```

## External Service Dependencies

```
CLOUD SERVICES:
├── OpenAI API (GPT-4o for translation)
├── Google Translate API
├── Google Gemini API (NEW)
├── Firebase (Authentication)
├── Redis (Message broker + caching)
├── YouTube (yt-dlp)
└── Render/Vercel (Deployment)

SYSTEM DEPENDENCIES:
├── FFmpeg (video processing)
├── Python 3.12+
├── Node.js 18+ (for frontend)
└── Docker (containerization)
```

## Data Flow

```
1. USER INPUT
   └─→ Frontend (React)
       └─→ HTTP Request to Backend

2. REQUEST PROCESSING (app.py)
   └─→ Validation
       └─→ File upload/URL parsing
           └─→ Queue Celery task

3. CELERY WORKER (tasks.py)
   ├─→ Download video (yt-dlp)
   ├─→ Extract audio (FFmpeg)
   ├─→ Transcribe (whisper_smart.py or gemini_transcription.py)
   ├─→ Translate (translation_services.py)
   ├─→ Create SRT (video_utils.py)
   ├─→ Validate (quality_gate.py)
   └─→ Embed/Process (video_utils.py)

4. RESULT STORAGE
   └─→ Redis/Database
       └─→ HTTP Response to Frontend

5. FRONTEND DISPLAY
   └─→ React component renders results
       └─→ Download link provided to user
```

## Testing Architecture

```
UNIT TESTS (backend/tests/unit/):
├── test_translation_services_unit.py
├── test_whisper_config_protection.py
├── test_metadata_service.py
├── test_video_utils_*.py (multiple)
├── test_segment_batching.py
├── test_openai_configuration.py
└── [+ 12 more unit tests]

INTEGRATION TESTS (backend/tests/integration/):
├── Various integration tests
└── Database/service integration

E2E TESTS:
├── e2e/ (Playwright)
│   ├── tests/smoke/
│   ├── tests/critical/
│   ├── tests/extended/
│   └── tests/regression/
│
└── Root level (scattered):
    ├── test_phase_a_integration.py
    ├── test_stats_jsonl.py
    ├── test_stats_upload.py
    ├── test_online_video.py
    └── test_production_youtube.py

REQUIREMENTS:
├── requirements.txt (production)
├── requirements-test.txt (testing)
└── constraints.txt (dependency constraints)
```

---
**Generated**: November 24, 2025
**Analysis Status**: COMPLETE
