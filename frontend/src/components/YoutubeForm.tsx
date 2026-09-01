import { useState, useEffect } from 'react';
import { useTranslation } from '../i18n/TranslationContext';
import type { WhisperModel, TranslationService, DownloadMediaFormat } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8081';

interface VideoCandidate {
  url: string;
  title: string;
  duration: number | null;
  duration_string: string;
  thumbnail: string;
  uploader: string;
}

interface YoutubeFormProps {
  onYoutubeSubmit: (
    youtubeUrl: string,
    sourceLang: string,
    targetLang: string,
    autoCreateVideo: boolean,
    whisperModel: WhisperModel,
    translationService: TranslationService
  ) => void;
  onQuickDownload: (url: string, mediaFormat: DownloadMediaFormat) => void;
  isProcessing: boolean;
  sourceLang: string;
  targetLang: string;
  autoCreateVideo: boolean;
  whisperModel: WhisperModel;
  translationService: TranslationService;
}

const YoutubeForm: React.FC<YoutubeFormProps> = ({
  onYoutubeSubmit,
  onQuickDownload,
  isProcessing,
  sourceLang,
  targetLang,
  autoCreateVideo,
  whisperModel,
  translationService
}) => {
  const { t } = useTranslation();
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [validationError, setValidationError] = useState('');
  const [isValidating, setIsValidating] = useState(false);
  const [downloadFormat, setDownloadFormat] = useState<DownloadMediaFormat>('mp4');
  // The Telegram-style instant preview: a YouTube id has a deterministic
  // thumbnail URL, and the title comes from one cheap key-less oEmbed call
  // proxied by the backend. Appears within ~a second of pasting, costs nothing.
  const [preview, setPreview] = useState<{ id: string; title?: string; author?: string } | null>(null);

  // Page-URL resolution + multi-video picker state
  const [resolving, setResolving] = useState(false);
  const [candidates, setCandidates] = useState<VideoCandidate[] | null>(null);
  // The action to run once a single video URL is known (process or download).
  const [pendingRun, setPendingRun] = useState<((url: string) => void) | null>(null);
  // Set when the server capped a long playlist, so the picker can say so
  // instead of silently pretending the list is all of it.
  const [candidatesTotal, setCandidatesTotal] = useState<number | null>(null);

  // Extract a YouTube video id from any of the common URL shapes.
  const extractYoutubeId = (url: string): string | null => {
    const match = url.match(
      /(?:youtube\.com\/(?:watch\?(?:.*&)?v=|shorts\/|embed\/)|youtu\.be\/)([A-Za-z0-9_-]{11})/
    );
    return match ? match[1] : null;
  };

  // Instant preview on paste (debounced with the validation below).
  useEffect(() => {
    const id = extractYoutubeId(youtubeUrl.trim());
    if (!id) {
      setPreview(null);
      return;
    }
    if (preview?.id === id) return;
    // Thumbnail renders immediately from the deterministic URL; the title
    // fills in when the oEmbed answer arrives.
    setPreview({ id });
    let cancelled = false;
    const timeoutId = setTimeout(() => {
      fetch(`${API_BASE_URL}/youtube-preview?video_id=${id}`)
        .then((r) => (r.ok ? r.json() : null))
        .then((data) => {
          if (!cancelled && data?.title) {
            setPreview({ id, title: data.title, author: data.author });
          }
        })
        .catch(() => {});
    }, 350);
    return () => {
      cancelled = true;
      clearTimeout(timeoutId);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [youtubeUrl]);

  // Validate URL in real-time with debounce
  useEffect(() => {
    if (!youtubeUrl.trim()) {
      setValidationError('');
      return;
    }

    setIsValidating(true);
    const timeoutId = setTimeout(() => {
      const error = validateVideoUrl(youtubeUrl);
      setValidationError(error);
      setIsValidating(false);
    }, 500); // 500ms debounce

    return () => clearTimeout(timeoutId);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [youtubeUrl]);

  const validateVideoUrl = (url: string): string => {
    const trimmedUrl = url.trim();
    if (!trimmedUrl) return '';

    try {
      const urlObj = new URL(trimmedUrl);
      const domain = urlObj.hostname.toLowerCase();

      // Popular supported domains (yt-dlp supports 1849+ sites)
      const popularDomains = [
        // YouTube
        'youtube.com', 'www.youtube.com', 'youtu.be', 'm.youtube.com', 'music.youtube.com',
        // Other popular video sites
        'vimeo.com', 'dailymotion.com', 'facebook.com', 'fb.watch',
        'instagram.com', 'tiktok.com', 'twitch.tv', 'reddit.com',
        'soundcloud.com', 'twitter.com', 'x.com', 'foxnews.com'
      ];

      // Check if it's a known popular domain
      const isKnownDomain = popularDomains.some(validDomain => domain.includes(validDomain));

      if (isKnownDomain) {
        // Additional validation for known domains
        if (domain.includes('youtube.com')) {
          if (!urlObj.pathname.includes('/watch') && !urlObj.searchParams.has('v') && !urlObj.pathname.includes('/shorts/')) {
            return t('validation.invalidYouTubeUrl');
          }
        } else if (domain === 'youtu.be') {
          if (urlObj.pathname.length <= 1) {
            return t('validation.invalidYouTubeUrl');
          }
        }
        return ''; // Valid
      } else {
        // For unknown domains (incl. arbitrary pages), let the server resolve it
        return ''; // Let server handle validation
      }
    } catch {
      return t('validation.invalidUrl');
    }
  };

  // Probe the URL server-side: a direct link runs immediately; a page with one
  // video runs immediately; a page with several videos opens the picker; a page
  // with no video shows a friendly message. Falls back to running the raw URL
  // if the probe itself fails (network), to preserve existing behavior.
  const resolveThenRun = async (url: string, run: (finalUrl: string) => void) => {
    setResolving(true);
    setValidationError('');
    try {
      const response = await fetch(`${API_BASE_URL}/resolve-url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
        credentials: 'include',
      });
      const data = await response.json();

      if (!response.ok) {
        setValidationError(data.error || t('videoPicker.unavailable'));
        return;
      }

      if (data.type === 'single' && data.video) {
        run(data.video.url || url);
      } else if (data.type === 'multiple' && Array.isArray(data.videos)) {
        setCandidates(data.videos);
        setCandidatesTotal(data.truncated ? data.total : null);
        setPendingRun(() => run);
      } else {
        // type === 'none'
        const reasonKey =
          data.reason === 'needs_login' ? 'videoPicker.needsLogin'
          : data.reason === 'unavailable' ? 'videoPicker.unavailable'
          : 'videoPicker.noVideo';
        setValidationError(t(reasonKey));
      }
    } catch {
      // Probe failed (e.g. network) — fall back to the original behavior.
      run(url);
    } finally {
      setResolving(false);
    }
  };

  const handlePick = (video: VideoCandidate) => {
    const run = pendingRun;
    setCandidates(null);
    setCandidatesTotal(null);
    setPendingRun(null);
    if (run) run(video.url);
  };

  const handleCancelPicker = () => {
    setCandidates(null);
    setCandidatesTotal(null);
    setPendingRun(null);
  };

  const handleSubmit = () => {
    const trimmedUrl = youtubeUrl.trim();
    if (!trimmedUrl) {
      setValidationError(t('validation.enterVideoUrl'));
      return;
    }

    const error = validateVideoUrl(trimmedUrl);
    if (error) {
      setValidationError(error);
      return;
    }

    resolveThenRun(trimmedUrl, (finalUrl) =>
      onYoutubeSubmit(finalUrl, sourceLang, targetLang, autoCreateVideo, whisperModel, translationService)
    );
  };

  const handleQuickDownload = () => {
    const trimmedUrl = youtubeUrl.trim();
    if (!trimmedUrl) {
      setValidationError(t('validation.enterVideoUrl'));
      return;
    }

    const error = validateVideoUrl(trimmedUrl);
    if (error) {
      setValidationError(error);
      return;
    }

    resolveThenRun(trimmedUrl, (finalUrl) => onQuickDownload(finalUrl, downloadFormat));
  };

  const busy = isProcessing || resolving;

  return (
    <div className="youtube-section">
      <div className="youtube-input-group">
        <div className="relative">
          <input
            type="text"
            className={`youtube-input transition-all duration-300 ${
              validationError
                ? 'border-red-500 border-2 bg-red-50 focus:border-red-600 focus:ring-red-200'
                : 'border-gray-300 focus:border-blue-500 focus:ring-blue-200'
            }`}
            placeholder={t('youtube.urlPlaceholder')}
            value={youtubeUrl}
            onChange={(e) => setYoutubeUrl(e.target.value)}
            disabled={busy}
            dir="ltr"
          />
          {isValidating && (
            <div className="absolute left-3 top-1/2 transform -translate-y-1/2">
              <div className="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>
            </div>
          )}
        </div>

        {preview && !validationError && (
          <div className="mt-2 p-2.5 bg-white border border-gray-200 rounded-xl flex items-center gap-3 animate-fadeIn" dir="rtl">
            <img
              src={`https://i.ytimg.com/vi/${preview.id}/hqdefault.jpg`}
              alt=""
              className="rounded-lg object-cover flex-shrink-0"
              style={{ width: '96px', height: '54px' }}
              onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
            />
            <div className="min-w-0 text-right">
              <div className="text-sm font-medium text-gray-800 truncate">
                {preview.title || t('videoPicker.resolving')}
              </div>
              {preview.author && (
                <div className="text-xs text-gray-500 truncate">{preview.author}</div>
              )}
            </div>
          </div>
        )}

        {validationError && (
          <div className="mt-2 p-3 bg-red-50 border border-red-200 rounded-xl text-red-700 text-sm animate-fadeIn" dir="rtl">
            <div className="flex items-start gap-2">
              {/* eslint-disable-next-line i18next/no-literal-string */}
              <span className="text-red-500 mt-0.5">⚠️</span>
              <span>{validationError}</span>
            </div>
          </div>
        )}

        {resolving && (
          <div className="mt-2 p-3 bg-blue-50 border border-blue-200 rounded-xl text-blue-700 text-sm flex items-center gap-2" dir="rtl">
            <div className="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>
            <span>{t('videoPicker.resolving')}</span>
          </div>
        )}

        <div className="youtube-buttons mt-4">
          <button
            onClick={handleSubmit}
            disabled={busy || !youtubeUrl.trim() || !!validationError}
            className="youtube-btn youtube-btn-process disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
          >
            {isProcessing ? (
              <div className="flex items-center gap-2">
                <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full"></div>
                <span>{t('buttons.processing')}</span>
              </div>
            ) : (
              <>📥 {t('buttons.processButton')}</>
            )}
          </button>
          <button
            onClick={handleQuickDownload}
            disabled={busy || !youtubeUrl.trim() || !!validationError}
            className="youtube-btn youtube-btn-download disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
          >
            {downloadFormat === 'mp3' ? '🎵' : '🎬'} {t('buttons.downloadOnly')}
          </button>
        </div>

        <div className="mt-3 flex items-center gap-2 flex-wrap" dir="rtl">
          <span className="text-sm text-gray-600">{t('youtube.downloadFormat')}</span>
          <div className="inline-flex rounded-xl border border-gray-300 overflow-hidden">
            {(['mp4', 'mp3'] as DownloadMediaFormat[]).map((format) => (
              <button
                key={format}
                type="button"
                onClick={() => setDownloadFormat(format)}
                disabled={busy}
                aria-pressed={downloadFormat === format}
                className={`px-3 py-1.5 text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                  downloadFormat === format
                    ? 'bg-blue-500 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-50'
                }`}
              >
                {format === 'mp4'
                  ? <>🎬 {t('youtube.downloadFormat_mp4')}</>
                  : <>🎵 {t('youtube.downloadFormat_mp3')}</>}
              </button>
            ))}
          </div>
        </div>
      </div>

      {candidates && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
          onClick={handleCancelPicker}
        >
          <div
            className="bg-white rounded-2xl shadow-xl max-w-lg w-full max-h-[80vh] overflow-y-auto p-5"
            dir="rtl"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-lg font-bold text-gray-800">{t('videoPicker.title')}</h3>
            <p className="text-sm text-gray-500 mt-1">{t('videoPicker.subtitle')}</p>
            {candidatesTotal !== null && (
              <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 mt-2">
                {t('videoPicker.truncated', { shown: candidates.length, total: candidatesTotal })}
              </p>
            )}
            <div className="mb-4" />

            <ul className="space-y-2">
              {candidates.map((video, idx) => (
                <li key={video.url || idx}>
                  <button
                    onClick={() => handlePick(video)}
                    className="w-full flex items-center gap-3 p-2 rounded-xl border border-gray-200 hover:border-blue-400 hover:bg-blue-50 transition-all text-right"
                  >
                    {video.thumbnail ? (
                      <img src={video.thumbnail} alt="" className="w-20 h-12 object-cover rounded-md flex-shrink-0" />
                    ) : (
                      // eslint-disable-next-line i18next/no-literal-string
                      <span className="w-20 h-12 rounded-md bg-gray-100 flex items-center justify-center flex-shrink-0">🎬</span>
                    )}
                    <span className="flex-1 min-w-0">
                      <span className="block text-sm font-medium text-gray-800 truncate">
                        {video.title || t('videoPicker.untitled')}
                      </span>
                      {video.duration_string && (
                        <span className="block text-xs text-gray-500">{video.duration_string}</span>
                      )}
                    </span>
                    <span className="text-xs text-blue-600 font-semibold flex-shrink-0">{t('videoPicker.useThis')}</span>
                  </button>
                </li>
              ))}
            </ul>

            <button
              onClick={handleCancelPicker}
              className="mt-4 w-full py-2 rounded-xl border border-gray-300 text-gray-600 hover:bg-gray-50 text-sm"
            >
              {t('videoPicker.cancel')}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default YoutubeForm;
