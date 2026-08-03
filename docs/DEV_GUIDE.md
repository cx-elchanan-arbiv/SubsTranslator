# SubsTranslator - Development Guide

## Development Environment Setup

### Prerequisites
- **Docker & Docker Compose** (recommended)
- **Python 3.12+** (for local development)
- **Node.js 16+** (for frontend development)
- **FFmpeg** (for video processing)
- **Git** (version control)

### Quick Start with Docker

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd SubsTranslator
   ```

2. **Start all services:**
   ```bash
   docker-compose up -d
   ```

3. **Access the application:**
   - Frontend: http://localhost
   - Backend API: http://localhost:8081
   - Redis: localhost:6379

4. **View logs:**
   ```bash
   docker-compose logs -f
   ```

### Local Development Setup

#### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set environment variables
export FLASK_ENV=development
export CELERY_BROKER_URL=redis://localhost:6379/0





# Start Flask development server
python app.py

# Start Celery worker (separate terminal)
celery -A celery_worker.celery worker --loglevel=info

# Start Celery beat (separate terminal)
celery -A celery_worker.celery beat --loglevel=info
```

#### Frontend Setup
```bash
cd frontend
npm install
npm start  # Starts development server on port 3000
```

## Project Structure

### Backend (`/backend/`)
```
backend/
├── app.py                    # Flask application entry point
├── config.py                 # Configuration classes
├── celery_config.py          # Celery configuration
├── celery_worker.py          # Celery worker setup
├── tasks.py                  # Celery task definitions
├── exceptions.py             # Custom exception hierarchy
├── logging_config.py         # Structured logging configuration
├── services/                 # Business logic services
│   └── subtitle_service.py   # Subtitle processing service
├── core/                     # Core utilities and shared code
├── requirements.txt          # Python dependencies
├── uploads/                  # Temporary file uploads
├── downloads/                # Processed output files
└── whisper_models/           # AI model cache directory
```

### Frontend (`/frontend/`)
```
frontend/
├── src/
│   ├── components/           # React components
│   ├── contexts/            # React contexts for state management
│   ├── hooks/               # Custom React hooks
│   ├── i18n/               # Internationalization files
│   ├── types/              # TypeScript type definitions
│   └── App.tsx             # Main application component
├── public/                  # Static assets
├── package.json            # Node.js dependencies
└── tailwind.config.js      # Tailwind CSS configuration
```

## Development Workflow

### Code Organization Conventions

#### File Naming
- **Services**: `*_service.py` (e.g., `subtitle_service.py`)
- **Utilities**: `*_utils.py` (e.g., `rtl_utils.py`)
- **Core modules**: `core/*.py`
- **Adapters**: `adapters/*_adapter.py` (for external services)

#### Class and Function Naming
- **Classes**: PascalCase (e.g., `SubtitleService`)
- **Functions**: snake_case (e.g., `create_srt_file`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_FILE_AGE`)

#### Import Organization
```python
# Standard library imports
import os
import json
from typing import List, Dict, Optional

# Third-party imports
import structlog
from flask import Flask

# Local imports
from config import get_config
from exceptions import VideoProcessingError
from services.subtitle_service import subtitle_service
```

### Testing Strategy

#### Running Tests
```bash
# Run all unit tests (fast)
python -m pytest -m "unit" -v

# Run integration tests
python -m pytest -m "integration" -v

# Run all tests except slow ones
python -m pytest -m "not slow" -v

# Run with coverage
python -m pytest --cov=backend --cov-report=html

# Run specific test file
python -m pytest tests/test_api_contracts.py -v
```

#### Test Categories
- **Unit Tests** (`@pytest.mark.unit`): Fast, isolated, no external dependencies
- **Integration Tests** (`@pytest.mark.integration`): Test component interactions
- **Slow Tests** (`@pytest.mark.slow`): Network-dependent or long-running tests

#### Test Structure
```python
import pytest
from unittest.mock import patch, MagicMock

@pytest.mark.unit
class TestSubtitleService:
    def setup_method(self):
        """Setup before each test method"""
        pass
    
    def test_create_srt_file_basic(self):
        """Test basic SRT file creation"""
        # Arrange
        segments = [{'start': 0.0, 'end': 2.5, 'text': 'Hello'}]
        
        # Act
        result = create_srt_file(segments, 'test.srt')
        
        # Assert
        assert result == 'test.srt'
```

#### Mocking External Services
```python
# Mock YouTube downloads
@patch('tasks.download_youtube_video')
def test_youtube_processing(mock_download):
    mock_download.return_value = ('/path/to/video.mp4', {})
    # Test implementation
```

### Code Quality Tools

#### Linting and Formatting
```bash
# Python linting with ruff
ruff check backend/
ruff format backend/

# Type checking with mypy
mypy backend/

# Frontend linting
cd frontend
npm run lint
npm run format
```

#### Pre-commit Hooks
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### Configuration Management

#### Environment Variables
```bash
# Development (.env file)
FLASK_ENV=development
USE_FAKE_YTDLP=true
LOG_LEVEL=DEBUG
JSON_LOGS=false

# Testing
TESTING=true
USE_FAKE_YTDLP=true

# Production
FLASK_ENV=production
USE_FAKE_YTDLP=false
LOG_LEVEL=INFO
JSON_LOGS=true
```

#### Configuration Classes
```python
class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    USE_FAKE_YTDLP = True
    
class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    USE_FAKE_YTDLP = True
    
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    USE_FAKE_YTDLP = False
```

## API Development

### Adding New Endpoints
1. **Define route in `app.py`:**
   ```python
   @app.route('/api/new-endpoint', methods=['POST'])
   def new_endpoint():
       try:
           # Validate input
           data = request.get_json()
           
           # Process request
           result = process_data(data)
           
           # Return response
           return jsonify({'status': 'success', 'result': result})
       except Exception as e:
           logger.error("Endpoint error", error=str(e))
           return jsonify({'status': 'error', 'message': str(e)}), 500
   ```

2. **Add corresponding Celery task if needed:**
   ```python
   @celery_app.task(bind=True)
   def new_processing_task(self, data):
       """Process data asynchronously"""
       try:
           # Task implementation
           return {'status': 'SUCCESS', 'result': result}
       except Exception as e:
           return {'status': 'FAILURE', 'error': str(e)}
   ```

3. **Write tests:**
   ```python
   @pytest.mark.integration
   def test_new_endpoint(client):
       response = client.post('/api/new-endpoint', 
                            json={'param': 'value'})
       assert response.status_code == 200
       assert response.json['status'] == 'success'
   ```

### API Response Format
```python
# Success response
{
    "status": "success",
    "result": {
        "data": "...",
        "metadata": "..."
    }
}

# Error response
{
    "status": "error", 
    "message": "Error description",
    "error_code": "SPECIFIC_ERROR_CODE"
}

# Task response
{
    "task_id": "uuid-string",
    "status": "PENDING|PROGRESS|SUCCESS|FAILURE",
    "result": {...}
}
```

## Frontend Development

### Component Development
```typescript
// TypeScript component with props interface
interface VideoUploadProps {
  onUpload: (file: File) => void;
  isUploading: boolean;
}

export const VideoUpload: React.FC<VideoUploadProps> = ({ 
  onUpload, 
  isUploading 
}) => {
  // Component implementation
};
```

### State Management
```typescript
// Custom hook for API calls
export const useApi = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const callApi = async (endpoint: string, data: any) => {
    setLoading(true);
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      return await response.json();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  return { callApi, loading, error };
};
```

### Internationalization
```typescript
// Using i18next for translations
import { useTranslation } from 'react-i18next';

export const Component = () => {
  const { t } = useTranslation();
  
  return (
    <div>
      <h1>{t('welcome.title')}</h1>
      <p>{t('welcome.description')}</p>
    </div>
  );
};
```

## Debugging

### Backend Debugging
```python
# Add debug logging
logger.debug("Debug info", extra_data=data)

# Use pdb for interactive debugging
import pdb; pdb.set_trace()

# Check task status
from celery_worker import celery
result = celery.AsyncResult('task-id')
print(result.status, result.result)
```

### Frontend Debugging
```typescript
// Console logging
console.log('Debug info:', data);

// React DevTools
// Install React Developer Tools browser extension

// Network debugging
// Use browser DevTools Network tab
```

### Docker Debugging
```bash
# View container logs
docker-compose logs -f backend
docker-compose logs -f worker

# Execute commands in running container
docker-compose exec backend bash
docker-compose exec worker python -c "import tasks; print('OK')"

# Check container status
docker-compose ps
docker stats
```

## Performance Optimization

### Backend Performance
- **Use appropriate Whisper model size** for speed vs accuracy tradeoff
- **Enable Redis caching** for repeated operations
- **Implement connection pooling** for database connections
- **Use async processing** for long-running tasks
- **Optimize FFmpeg parameters** for faster video processing

### Frontend Performance
- **Code splitting** with React.lazy()
- **Memoization** with React.memo() and useMemo()
- **Optimize bundle size** with webpack-bundle-analyzer
- **Lazy loading** for large components
- **Image optimization** for static assets

### Database/Cache Optimization
- **Redis key expiration** for temporary data
- **Efficient task queuing** with Celery routing
- **File cleanup scheduling** to prevent disk space issues

## Deployment

### Docker Production Build
```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy with production configuration
docker-compose -f docker-compose.prod.yml up -d

# Health check
curl http://localhost/health
```

### Environment Configuration
```bash
# Production environment variables
FLASK_ENV=production
DEBUG=false
USE_FAKE_YTDLP=false
LOG_LEVEL=INFO
JSON_LOGS=true
CELERY_WORKER_CONCURRENCY=4
```

## Troubleshooting

### Common Issues

#### "ModuleNotFoundError" in tests
```bash
# Ensure backend is in Python path
export PYTHONPATH="${PYTHONPATH}:./backend"
# Or use pytest from backend directory
cd backend && python -m pytest ../tests/
```

#### Celery worker not processing tasks
```bash
# Check Redis connection
redis-cli ping

# Check Celery worker status
celery -A celery_worker.celery inspect active

# Restart worker
docker-compose restart worker
```

#### FFmpeg errors
```bash
# Check FFmpeg installation
ffmpeg -version

# Check file permissions
ls -la uploads/ downloads/

# Enable debug logging
export LOG_LEVEL=DEBUG
```

#### Frontend build issues
```bash
# Clear node modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Check for TypeScript errors
npm run type-check
```

### Performance Issues
- **Monitor container resources**: `docker stats`
- **Check disk space**: `df -h`
- **Monitor Redis memory**: `redis-cli info memory`
- **Profile Python code**: Use `cProfile` or `py-spy`

## Contributing Guidelines

### Pull Request Process
1. **Create feature branch** from main
2. **Write tests** for new functionality
3. **Run full test suite** and ensure all tests pass
4. **Update documentation** if needed
5. **Submit PR** with clear description

### Code Review Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact considered
- [ ] Security implications reviewed

### Commit Message Format
```
type(scope): description

feat(api): add new video processing endpoint
fix(frontend): resolve upload progress display issue
docs(readme): update installation instructions
test(backend): add unit tests for subtitle service
```

---

*Last updated: December 2024*
