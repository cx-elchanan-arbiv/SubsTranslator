import { useRef, useState, useEffect } from 'react';
import { useTranslation } from '../i18n/TranslationContext';

interface FileMetadata {
  duration?: number;
  width?: number;
  height?: number;
}

interface UploadFormProps {
  selectedFile: File | null;
  onFileSelect: (file: File | null) => void;
  isProcessing: boolean;
}

const UploadForm: React.FC<UploadFormProps> = ({ selectedFile, onFileSelect, isProcessing }) => {
  const { t } = useTranslation();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [fileMetadata, setFileMetadata] = useState<FileMetadata>({});
  // A real frame from the selected video (data URL), shown instead of the
  // generic 📄 — the same preview Finder shows. Extracted locally in the
  // browser: no upload, no server, no cost.
  const [thumbnail, setThumbnail] = useState<string | null>(null);

  // Extract video metadata when file is selected
  useEffect(() => {
    if (!selectedFile) {
      setFileMetadata({});
      setThumbnail(null);
      return;
    }

    // Only extract metadata for video files
    const videoExtensions = ['mp4', 'mkv', 'mov', 'webm', 'avi'];
    const fileExtension = selectedFile.name.split('.').pop()?.toLowerCase();

    if (!fileExtension || !videoExtensions.includes(fileExtension)) {
      setFileMetadata({});
      setThumbnail(null);
      return;
    }

    // Create video element to read metadata + capture one frame
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.muted = true;
    const objectUrl = URL.createObjectURL(selectedFile);
    let done = false;

    const cleanup = () => {
      if (!done) {
        done = true;
        URL.revokeObjectURL(objectUrl);
      }
    };

    video.onloadedmetadata = () => {
      setFileMetadata({
        duration: video.duration,
        width: video.videoWidth,
        height: video.videoHeight,
      });
      // Seek to a representative frame: 10% in, but between 0.5s and 3s —
      // frame 0 is very often black or a fade-in.
      try {
        video.currentTime = Math.min(Math.max(video.duration * 0.1, 0.5), 3);
      } catch {
        cleanup();
      }
    };

    video.onseeked = () => {
      try {
        const canvas = document.createElement('canvas');
        const scale = Math.min(1, 320 / (video.videoWidth || 320));
        canvas.width = Math.round((video.videoWidth || 320) * scale);
        canvas.height = Math.round((video.videoHeight || 180) * scale);
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          setThumbnail(canvas.toDataURL('image/jpeg', 0.8));
        }
      } catch {
        // Frame capture is a nicety — the icon fallback is fine.
      }
      cleanup();
    };

    video.onerror = () => {
      // If video fails to load, just skip metadata
      setFileMetadata({});
      setThumbnail(null);
      cleanup();
    };

    video.src = objectUrl;
    return cleanup;
  }, [selectedFile]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onFileSelect(file);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const formatDuration = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  };

  const getFileExtension = (filename: string): string => {
    return filename.split('.').pop()?.toUpperCase() || '';
  };

  return (
    <div className="upload-section">
      {!selectedFile ? (
        <>
          <div
            className={`upload-area ${isProcessing ? 'disabled' : ''}`}
            onClick={() => !isProcessing && fileInputRef.current?.click()}
          >
            <div className="upload-icon">📎</div>
            <p className="upload-text">{t.uploadTitle}</p>
            <small className="upload-hint">{t.supportedFormats}</small>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".mp4,.mkv,.mov,.webm,.avi,.mp3,.wav"
            onChange={handleFileChange}
            style={{ display: 'none' }}
            disabled={isProcessing}
          />
        </>
      ) : (
        <div className="file-preview-container">
          <div className="file-preview">
            {thumbnail ? (
              <img
                src={thumbnail}
                alt=""
                className="file-icon rounded-lg object-cover"
                style={{ width: '96px', height: '54px' }}
              />
            ) : (
              <div className="file-icon">📄</div>
            )}
            <div className="file-details">
              <div className="file-name">{selectedFile.name}</div>
              <div className="file-meta-grid">
                {fileMetadata.duration && (
                  <div className="file-meta-item">
                    <span className="meta-icon">⏱️</span>
                    <span className="meta-label">{t('upload.duration') || 'משך'}:</span>
                    <span className="meta-value">{formatDuration(fileMetadata.duration)}</span>
                  </div>
                )}
                <div className="file-meta-item">
                  <span className="meta-icon">📊</span>
                  <span className="meta-label">{t('upload.fileSize') || 'גודל קובץ'}:</span>
                  <span className="meta-value">{formatFileSize(selectedFile.size)}</span>
                </div>
                {fileMetadata.width && fileMetadata.height && (
                  <div className="file-meta-item">
                    <span className="meta-icon">🎬</span>
                    <span className="meta-label">{t('upload.resolution') || 'רזולוציה'}:</span>
                    <span className="meta-value">{fileMetadata.width}×{fileMetadata.height}</span>
                  </div>
                )}
              </div>
            </div>
            <button
              className="remove-file-btn"
              onClick={() => {
                onFileSelect(null);
                if (fileInputRef.current) {
                  fileInputRef.current.value = '';
                }
              }}
              disabled={isProcessing}
              title={t.removeFile || 'Remove file'}
            >
              ✕
            </button>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".mp4,.mkv,.mov,.webm,.avi,.mp3,.wav"
            onChange={handleFileChange}
            style={{ display: 'none' }}
            disabled={isProcessing}
          />
        </div>
      )}
    </div>
  );
};

export default UploadForm;
