# SubsTranslator - Project Overview

## What is SubsTranslator?

SubsTranslator is an advanced AI-powered video subtitle generation and translation system. It automatically transcribes video content using state-of-the-art speech recognition and translates subtitles into multiple languages with specialized support for Hebrew and other RTL languages.

## Key Features

### Core Functionality
- **⚡ Ultra-Fast Transcription**: Uses `faster-whisper` with intelligent model selection (tiny/base/medium/large-v3)
- **🌍 Multi-language Support**: 11+ languages with specialized Hebrew/Arabic RTL processing
- **📱 Dual Input Methods**: YouTube URL processing + local file upload support
- **🔄 Asynchronous Processing**: Celery + Redis for scalable background task processing

### Advanced Subtitle Features
- **🔥 Burn-in Subtitles**: Create videos with embedded subtitles using FFmpeg
- **📝 Manual Subtitle Embedding**: Direct subtitle burn-in from text input with timestamp support
- **🎨 Hebrew Text Optimization**: Sophisticated RTL text handling with proper directional markers
- **⚙️ Custom Styling**: Configurable subtitle appearance with Hebrew font optimization

### Professional Features
- **🖼️ Watermark System**: Automatic logo overlay with customizable positioning
- **⬇️ Quick Download Mode**: YouTube video download without processing
- **📊 Real-time Progress**: Live processing updates with detailed status information
- **🗂️ Multiple Output Formats**: Original SRT, translated SRT, and video with subtitles
- **🧹 Automated Cleanup**: Background file management and cleanup tasks

## How It Works

### Basic Workflow
1. **Input**: User provides YouTube URL or uploads video file
2. **Download**: System downloads/processes the video file
3. **Transcription**: AI extracts speech and converts to text using Whisper models
4. **Translation**: Text is translated to target language using Google Translate or other services
5. **Subtitle Creation**: Generate SRT files with proper formatting and RTL support
6. **Video Processing**: Optionally embed subtitles into video with watermark
7. **Output**: User receives SRT files and/or processed video

### Supported Languages
- **Source Languages**: Auto-detection or manual selection
- **Target Languages**: Hebrew, English, Arabic, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese
- **Special RTL Support**: Hebrew, Arabic, Farsi, Urdu, Yiddish

### Output Formats
- **Original SRT**: Transcribed text in source language
- **Translated SRT**: Translated subtitles in target language  
- **Video with Subtitles**: MP4 with burned-in subtitles and watermark
- **Download-only**: Raw video file without processing

## Technology Stack

### Backend
- **Python 3.12+** with Flask web framework
- **Celery** for asynchronous task processing
- **Redis** as message broker and cache
- **faster-whisper** for AI transcription
- **FFmpeg** for video processing
- **yt-dlp** for YouTube video downloading

### Frontend
- **React 18** with TypeScript
- **Tailwind CSS** for styling
- **Framer Motion** for animations
- **i18next** for internationalization (Hebrew/English)

### Infrastructure
- **Docker & Docker Compose** for containerization
- **Nginx** as reverse proxy
- **Gunicorn** as WSGI server
- **File-based storage** with automatic cleanup

## Use Cases

### Content Creators
- Generate subtitles for YouTube videos
- Translate content for international audiences
- Create accessible content with proper RTL support

### Educational Institutions
- Transcribe lectures and educational content
- Provide multilingual subtitles for diverse student populations
- Create searchable video content archives

### Media Companies
- Automate subtitle generation for video content
- Localize content for different markets
- Ensure accessibility compliance

### Developers & Researchers
- Process video datasets for machine learning
- Extract and analyze speech content from videos
- Build upon the subtitle generation pipeline

## Getting Started

### Quick Start (Docker)
```bash
git clone <repository-url>
cd SubsTranslator
docker-compose up -d
```

Access the application at `http://localhost`

### Manual Setup
See `DEV_GUIDE.md` for detailed development setup instructions.

## Project Status

The project is actively maintained and includes:
- ✅ **Stable Core Features**: Transcription, translation, subtitle generation
- ✅ **Production Ready**: Docker deployment, error handling, logging
- ✅ **Modern Architecture**: Clean code structure with services and proper separation of concerns
- 🔄 **Continuous Improvement**: Regular updates to AI models and processing capabilities

## Support & Documentation

- **Architecture Details**: See `ARCHITECTURE.md`
- **Development Guide**: See `DEV_GUIDE.md`
- **API Documentation**: Available in `/docs/api/`
- **Issue Tracking**: GitHub Issues

## License

[Add license information here]

---

*Last updated: December 2024*
