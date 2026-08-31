
import { useState } from 'react';
import { motion } from 'framer-motion';
import { VideoMetadata, FileMetadata, UserChoices, Progress } from '../hooks/useApi';
import { useTranslation } from '../i18n/TranslationContext';
import AILoader from './AILoader';


interface ProgressDisplayProps {
  isProcessing: boolean;
  progress: Progress;
  error: string | { error_code?: string; message?: string; user_facing_message?: string } | null;
  videoMetadata?: VideoMetadata;
  fileMetadata?: FileMetadata;
  userChoices?: UserChoices;
  initialRequest?: any;
  processingType?: 'file_upload' | 'youtube_full' | 'youtube_download' | null;
  onRetry?: () => void;
  /** Files that survived a failed run — the task marks these `salvaged`. */
  salvagedResult?: { files?: Record<string, string> } | null;
}

interface EnhancedStep {
  id: number;
  label: string;
  status: 'pending' | 'in_progress' | 'completed' | 'error';
  weight: number;
  progress: number;
}

// Maps the English labels emitted by the backend to the camelCase keys
// used in the frontend locale files under processingSteps.*
const BACKEND_LABEL_TO_I18N_KEY: Record<string, string> = {
  'Downloading': 'downloadingVideo',
  'Downloading Video': 'downloadingVideo',
  'Audio Processing': 'audioProcessing',
  'Loading AI Model': 'modelLoading',
  'AI Transcription': 'transcription',
  'Translation': 'translation',
  'SRT Creation': 'srtCreation',
  'Creating Subtitle Files': 'srtCreation',
  'Video Creation': 'videoCreation',
  'Embedding Subtitles': 'videoCreation',
  'Finalizing Video': 'finalizing',
};

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8081';

const ProgressDisplay: React.FC<ProgressDisplayProps> = ({ isProcessing, progress, error, videoMetadata, fileMetadata, userChoices, initialRequest, processingType, onRetry, salvagedResult }) => {
  const { t } = useTranslation();
  const [showLogs, setShowLogs] = useState(false);
  const subtitlePositionLabel = userChoices?.subtitle_position
    ? t(`options.subtitlePosition${userChoices.subtitle_position[0].toUpperCase()}${userChoices.subtitle_position.slice(1)}`)
    : '';
  const subtitlePositionIcon = String.fromCodePoint(0x2195, 0xfe0f);

  if (!isProcessing && !error) return null;

  // For YouTube processing, show AI loader with context while waiting for video metadata
  const isYouTubeProcessing = processingType === 'youtube_full' || processingType === 'youtube_download';
  if (isYouTubeProcessing && !videoMetadata && !error && progress.overall_percent < 10) {
    // Show immediate context if we have initialRequest or userChoices
    const hasContext = (userChoices && Object.keys(userChoices).length > 0) ||
                      (initialRequest && Object.keys(initialRequest).length > 0);

    if (hasContext) {
      return (
        <div className="max-w-4xl mx-auto space-y-4" dir="rtl">
          <AILoader message={t('processing.analyzingYoutube')} />

          {/* Show immediate context */}
          <div className="bg-blue-50 rounded-xl p-4 border border-blue-200">
            <div className="flex items-center gap-2 mb-3">
              {/* eslint-disable-next-line i18next/no-literal-string */}
              <span className="text-lg">⚙️</span>
              <span className="font-medium text-blue-700">{t('processing.requestDetails')}</span>
            </div>

            {userChoices && (
              <div className="grid grid-cols-2 gap-3 text-sm text-gray-600 mb-2">
                <div>🔤 {t('language.source')}: {t.languages[userChoices.source_lang] || userChoices.source_lang}</div>
                <div>🔄 {t('language.target')}: {t.languages[userChoices.target_lang] || userChoices.target_lang}</div>
                <div>🎯 {t('options.whisperModel')}: {userChoices.whisper_model}</div>
                <div>📹 {t('options.autoCreateVideo')}: {userChoices.auto_create_video ? '✅ ' + t('actions.yes') : '❌ ' + t('actions.no')}</div>
                {userChoices.auto_create_video && subtitlePositionLabel && (
                  <div>{subtitlePositionIcon} {t('options.subtitlePosition')}: {subtitlePositionLabel}</div>
                )}
              </div>
            )}

            {initialRequest && (
              <div className="text-sm text-gray-600">
                <div>📋 {t('processing.actionType')}: {initialRequest.type === 'download_only' ? t('processing.downloadOnly') : t('processing.fullProcessing')}</div>
                <div>🎬 {t('processing.quality')}: {initialRequest.quality}</div>
              </div>
            )}
          </div>
        </div>
      );
    }

    return (
      <div className="max-w-4xl mx-auto">
        <AILoader message={t('processing.analyzingYoutube')} />
      </div>
    );
  }

  // For file upload, show AI loader while waiting for processing steps to initialize
  // This prevents showing "0 out of 0 steps" during the initial fetch and first poll
  if (processingType === 'file_upload' && progress.steps.length === 0 && !error) {
    return (
      <div className="max-w-4xl mx-auto">
        <AILoader message={t('processing.preparingFile')} />
      </div>
    );
  }

  const getProcessingType = () => {
    if (processingType) return processingType;
    if (videoMetadata?.url) {
      const hasTranslationSteps = progress.steps?.some(step => 
        step.label.includes('translation') || step.label.includes('transcription')
      );
      return hasTranslationSteps ? 'youtube_full' : 'youtube_download';
    }
    return 'file_upload';
  };

  const currentProcessingType = getProcessingType();

  // Enhanced steps with weights and proper translation
  const getEnhancedSteps = (): EnhancedStep[] => {
    if (currentProcessingType === 'youtube_full' && progress.steps.length === 2) {
      // Create 7 estimated steps for YouTube full processing
      const estimatedSteps: EnhancedStep[] = [
        { id: 1, label: t('processingSteps.downloadingVideo'), status: 'pending', weight: 15, progress: 0 },
        { id: 2, label: t('processingSteps.audioProcessing'), status: 'pending', weight: 10, progress: 0 },
        { id: 3, label: t('processingSteps.modelLoading'), status: 'pending', weight: 5, progress: 0 },
        { id: 4, label: t('processingSteps.transcription'), status: 'pending', weight: 40, progress: 0 },
        { id: 5, label: t('processingSteps.translation'), status: 'pending', weight: 15, progress: 0 },
        { id: 6, label: t('processingSteps.srtCreation'), status: 'pending', weight: 10, progress: 0 },
        { id: 7, label: t('processingSteps.videoCreation'), status: 'pending', weight: 5, progress: 0 },
      ];

      const overallPercent = progress.overall_percent || 0;
      
      // Map progress to steps based on percentage thresholds
      const thresholds = [15, 25, 30, 70, 85, 95, 100];
      
      estimatedSteps.forEach((step, index) => {
        const threshold = thresholds[index];
        const prevThreshold = index > 0 ? thresholds[index - 1] : 0;
        
        if (overallPercent >= threshold) {
          step.status = 'completed';
          step.progress = 1;
        } else if (overallPercent > prevThreshold) {
          step.status = 'in_progress';
          step.progress = Math.min(1, (overallPercent - prevThreshold) / (threshold - prevThreshold));
        }
      });

      return estimatedSteps;
    }

    // For YouTube download, use the actual steps from backend
    if (currentProcessingType === 'youtube_download' && progress.steps.length > 0) {
      return progress.steps.map((step, index) => ({
        id: index + 1,
        label: t.steps[step.label] || step.label,
        status: step.status === 'waiting' ? 'pending' : step.status,
        weight: step.weight || 100 / progress.steps.length,
        progress: step.progress / 100
      }));
    }

    // Convert actual steps to enhanced steps
    return progress.steps.map((step, index) => {
      const i18nKey = BACKEND_LABEL_TO_I18N_KEY[step.label];
      const translated = i18nKey ? t(`processingSteps.${i18nKey}`) : null;
      return {
        id: index + 1,
        label: translated && translated !== `processingSteps.${i18nKey}` ? translated : step.label,
        status: step.status === 'waiting' ? 'pending' : step.status,
        weight: step.weight || 100 / progress.steps.length,
        progress: step.progress / 100
      };
    });
  };

  const enhancedSteps = getEnhancedSteps();
  const completedCount = enhancedSteps.filter(s => s.status === 'completed').length;
  const currentStep = enhancedSteps.find(s => s.status === 'in_progress');

  const getStepIcon = (status: string) => {
    switch (status) {
      case 'completed': return '✅';
      case 'in_progress': return '⚡';
      case 'error': return '❌';
      default: return '⏳';
    }
  };

  const getProcessingTitle = () => {
    switch (currentProcessingType) {
      case 'file_upload': return `📁 ${t('processing.processingFile')}`;
      case 'youtube_full': return `📺 ${t('processing.processingVideo')}`;
      case 'youtube_download': return `⬇️ ${t('processing.downloadingVideo')}`;
      default: return `🔄 ${t('status.processing')}`;
    }
  };

  if (error) {
    // Check if error is YouTube bot detection (structured error with code)
    const isYoutubeBotDetection = typeof error === 'object' &&
                                    (error as any)?.code === 'YOUTUBE_BOT_DETECTION';

    // A failure is only useful if it says what happened and what to do about it.
    // The backend classifies the cause into a stable code; the wording lives in the
    // locale files so it comes out in the user's language rather than in whatever
    // language the server happened to build the sentence in.
    const errorCode = typeof error === 'object' ? (error as any)?.code : undefined;
    const codedKey = errorCode ? `errors.byCode.${errorCode}` : '';
    // A missing key comes back as the key itself, which is the same test used for
    // the step labels above.
    const codedTitle =
      codedKey && t(`${codedKey}.title`) !== `${codedKey}.title` ? t(`${codedKey}.title`) : '';
    const codedBody = codedTitle ? t(`${codedKey}.body`) : '';
    const codedAction = codedTitle ? t(`${codedKey}.action`) : '';
    // The raw server message, shown small under the coded card. This is what
    // distinguishes "disk full" from "font missing" under one RENDER_FAILED
    // code, and carries the exact size limit under FILE_TOO_LARGE — the coded
    // copy explains, the detail line is the evidence.
    const technicalDetail =
      codedTitle && typeof error === 'object' && (error as any)?.message
        ? String((error as any).message)
        : '';

    // Files that survived the failure — the transcript is expensive and is already
    // on disk, so it is offered rather than thrown away.
    const salvagedFiles: string[] = Object.values(
      (salvagedResult?.files ?? {}) as Record<string, string>
    ).filter((name): name is string => typeof name === 'string' && name.length > 0);

    // "Try again" is hidden where the backend says a retry cannot succeed — an empty
    // account or a rejected key fails identically every time, and offering the button
    // invites the user to pay for another transcription for the same error.
    const retryWorthTrying = typeof error === 'object' && (error as any)?.recoverable === false
      ? false
      : true;

    if (isYoutubeBotDetection) {
      return (
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ duration: 0.4, ease: "easeOut" }}
          className="max-w-4xl mx-auto p-8 bg-white/80 backdrop-blur-lg rounded-2xl shadow-xl border border-blue-200/50"
          dir="rtl"
        >
          <div className="text-center">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
              className="text-6xl mb-6"
            >
              🚫
            </motion.div>
            <motion.h2
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="text-2xl font-bold text-blue-700 mb-4"
            >
              {t('errors.youtubeBotDetection.title')}
            </motion.h2>
            <motion.p
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.35 }}
              className="text-lg text-gray-700 mb-8"
            >
              {t('errors.youtubeBotDetection.message')}
            </motion.p>

            {/* Alternative Solution Box */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="bg-gradient-to-r from-blue-50 to-indigo-50 backdrop-blur border border-blue-200 rounded-xl p-6 mb-8 shadow-inner text-right"
            >
              <h3 className="text-xl font-bold text-blue-800 mb-4">
                {t('errors.youtubeBotDetection.alternativeTitle')}
              </h3>
              <div className="space-y-3 text-gray-700 text-base">
                <div className="flex items-start gap-3">
                  {/* eslint-disable-next-line i18next/no-literal-string */}
                  <span className="text-2xl flex-shrink-0">📥</span>
                  <p className="text-right flex-1">{t('errors.youtubeBotDetection.step1')}</p>
                </div>
                <div className="flex items-start gap-3">
                  {/* eslint-disable-next-line i18next/no-literal-string */}
                  <span className="text-2xl flex-shrink-0">↩️</span>
                  <p className="text-right flex-1">{t('errors.youtubeBotDetection.step2')}</p>
                </div>
                <div className="flex items-start gap-3">
                  {/* eslint-disable-next-line i18next/no-literal-string */}
                  <span className="text-2xl flex-shrink-0">✨</span>
                  <p className="text-right flex-1">{t('errors.youtubeBotDetection.step3')}</p>
                </div>
              </div>
            </motion.div>

            {/* Switch to File Upload Button */}
            <motion.button
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => {
                // Reset state first to clear the error
                if (onRetry) onRetry();
                // Then switch to upload tab
                const uploadTab = document.querySelector('[data-tab="upload"]') as HTMLElement;
                if (uploadTab) uploadTab.click();
              }}
              className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-medium py-3 px-8 rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl"
            >
              📁 {t('errors.youtubeBotDetection.switchButton')}
            </motion.button>
          </div>
        </motion.div>
      );
    }

    // Regular error display (unchanged)
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="max-w-4xl mx-auto p-8 bg-white/80 backdrop-blur-lg rounded-2xl shadow-xl border border-red-200/50"
        dir="rtl"
      >
        <div className="text-center">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
            className="text-5xl mb-6"
          >
            ❌
          </motion.div>
          <motion.h2
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="text-2xl font-bold text-red-600 mb-6"
          >
            {codedTitle || t('errors.processingError')}
          </motion.h2>
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="bg-gradient-to-r from-red-50 to-red-100/80 backdrop-blur border border-red-200 rounded-xl p-6 text-red-800 mb-8 shadow-inner"
          >
            <div className="flex items-start gap-3">
              {/* eslint-disable-next-line i18next/no-literal-string */}
              <span className="text-red-500 text-lg mt-1">⚠️</span>
              <div className="flex-1">
                {codedTitle ? (
                  <>
                    <p className="text-lg leading-relaxed">{codedBody}</p>
                    <p className="text-lg leading-relaxed mt-3 font-medium">{codedAction}</p>
                    {technicalDetail && technicalDetail !== codedBody && (
                      <p className="mt-3 font-mono text-xs text-red-700/80 bg-red-50 p-2 rounded" dir="ltr">
                        {technicalDetail}
                      </p>
                    )}
                  </>
                ) : (
                  (typeof error === 'string'
                    ? error
                    : (error as any)?.user_facing_message || (error as any)?.message || JSON.stringify(error)
                  ).split('\n').map((line, index) => (
                    <p key={index} className={`text-lg leading-relaxed ${index > 0 ? 'mt-2 font-mono text-sm bg-red-50 p-2 rounded' : ''}`}>
                      {line}
                    </p>
                  ))
                )}
              </div>
            </div>
          </motion.div>
          {salvagedFiles.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.45 }}
              className="bg-emerald-50 border border-emerald-200 rounded-xl p-5 mb-8 text-emerald-900"
            >
              <p className="font-semibold mb-3">{t('errors.salvageTitle')}</p>
              <div className="flex flex-wrap gap-3 justify-center">
                {salvagedFiles.map((filename) => (
                  <a
                    key={filename}
                    href={`${API_BASE_URL}/download/${encodeURIComponent(filename)}`}
                    download
                    className="bg-emerald-600 hover:bg-emerald-700 text-white font-medium py-2 px-5 rounded-lg transition-colors"
                  >
                    {/* eslint-disable-next-line i18next/no-literal-string */}
                    <span>⬇ </span>
                    {t('errors.salvageDownload')}
                  </a>
                ))}
              </div>
            </motion.div>
          )}
          {onRetry && retryWorthTrying && (
            <motion.button
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onRetry}
              className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-medium py-3 px-8 rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl"
            >
              🔄 {t('buttons.retryButton')}
            </motion.button>
          )}
        </div>
      </motion.div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6" dir="rtl">
      {/* Video Info Card (YouTube/Online Video) */}
      {videoMetadata && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gradient-to-l from-blue-50 to-indigo-50 rounded-2xl shadow-lg border border-blue-100 overflow-hidden"
        >
          <div className="p-6">
            <div className="flex items-start gap-6">
              {/* Thumbnail */}
              <div className="flex-shrink-0">
                <img
                  src={videoMetadata.thumbnail}
                  alt={videoMetadata.title}
                  className="w-32 h-24 rounded-lg object-cover shadow-md"
                />
              </div>

              {/* Video Info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">📺</span>
                  <span className="text-sm font-medium text-blue-600 bg-blue-100 px-2 py-1 rounded-full">
                    {t('processing.videoInfo')}
                  </span>
                </div>

                <h3 className="font-bold text-lg text-gray-800 mb-2 line-clamp-2">
                  {videoMetadata.title}
                </h3>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm text-gray-600 mb-3">
                  {/* eslint-disable-next-line i18next/no-literal-string */}
                  <div>⏱️ {t('processing.duration')}: {videoMetadata.duration_string}</div>
                  {/* eslint-disable-next-line i18next/no-literal-string */}
                  <div>👁️ {t('processing.views')}: {(videoMetadata.view_count || 0).toLocaleString()}</div>
                  <div>👤 {t('processing.creator')}: {videoMetadata.uploader}</div>
                  <div>🎬 {t('processing.resolution')}: {videoMetadata.width}×{videoMetadata.height}</div>
                </div>
              </div>
            </div>

            {/* User Settings */}
            {userChoices && (
              <div className="mt-4 pt-4 border-t border-blue-200">
                <div className="flex items-center gap-2 mb-3">
                  {/* eslint-disable-next-line i18next/no-literal-string */}
                  <span className="text-lg">⚙️</span>
                  <span className="text-sm font-medium text-blue-600 bg-blue-100 px-2 py-1 rounded-full">
                    {t('processing.yourChoices')}
                  </span>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm text-gray-600">
                  <div>🔤 {t('language.source')}: {t.languages[userChoices.source_lang] || userChoices.source_lang}</div>
                  <div>🔄 {t('language.target')}: {t.languages[userChoices.target_lang] || userChoices.target_lang}</div>
                  <div>🎯 {t('options.whisperModel')}: {userChoices.whisper_model}</div>
                  <div>📹 {t('options.autoCreateVideo')}: {userChoices.auto_create_video ? '✅ ' + t('actions.yes') : '❌ ' + t('actions.no')}</div>
                  {userChoices.auto_create_video && subtitlePositionLabel && (
                    <div>{subtitlePositionIcon} {t('options.subtitlePosition')}: {subtitlePositionLabel}</div>
                  )}
                </div>
              </div>
            )}
          </div>
        </motion.div>
      )}

      {/* File Info Card (Uploaded File) */}
      {fileMetadata && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gradient-to-l from-blue-50 to-indigo-50 rounded-2xl shadow-lg border border-blue-100 overflow-hidden"
        >
          <div className="p-6">
            <div className="flex items-start gap-6">
              {/* File Icon */}
              <div className="flex-shrink-0">
                <div className="w-32 h-24 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 shadow-md flex items-center justify-center">
                  <span className="text-5xl">📁</span>
                </div>
              </div>

              {/* File Info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">📄</span>
                  <span className="text-sm font-medium text-blue-600 bg-blue-100 px-2 py-1 rounded-full">
                    {t('processing.fileInfo') || 'מידע על הקובץ'}
                  </span>
                </div>

                <h3 className="font-bold text-lg text-gray-800 mb-2 line-clamp-2" title={fileMetadata.filename}>
                  {fileMetadata.filename}
                </h3>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm text-gray-600 mb-3">
                  {/* eslint-disable-next-line i18next/no-literal-string */}
                  <div>⏱️ {t('processing.duration')}: {fileMetadata.duration_string}</div>
                  <div>📊 {t('fileInfo.fileSizeLabel') || 'גודל'}: {fileMetadata.file_size_mb} MB</div>
                  <div>🎬 {t('processing.resolution')}: {fileMetadata.width}×{fileMetadata.height}</div>
                  {/* eslint-disable-next-line i18next/no-literal-string */}
                  <div>🎞️ FPS: {fileMetadata.fps}</div>
                </div>
              </div>
            </div>

            {/* User Settings */}
            {userChoices && (
              <div className="mt-4 pt-4 border-t border-blue-200">
                <div className="flex items-center gap-2 mb-3">
                  {/* eslint-disable-next-line i18next/no-literal-string */}
                  <span className="text-lg">⚙️</span>
                  <span className="text-sm font-medium text-blue-600 bg-blue-100 px-2 py-1 rounded-full">
                    {t('processing.yourChoices')}
                  </span>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm text-gray-600">
                  <div>🔤 {t('language.source')}: {t.languages[userChoices.source_lang] || userChoices.source_lang}</div>
                  <div>🔄 {t('language.target')}: {t.languages[userChoices.target_lang] || userChoices.target_lang}</div>
                  <div>🎯 {t('options.whisperModel')}: {userChoices.whisper_model}</div>
                  <div>📹 {t('options.autoCreateVideo')}: {userChoices.auto_create_video ? '✅ ' + t('actions.yes') : '❌ ' + t('actions.no')}</div>
                  {userChoices.auto_create_video && subtitlePositionLabel && (
                    <div>{subtitlePositionIcon} {t('options.subtitlePosition')}: {subtitlePositionLabel}</div>
                  )}
                </div>
              </div>
            )}
          </div>
        </motion.div>
      )}

      {/* Progress Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-2xl shadow-lg border border-gray-200 overflow-hidden"
      >
        <div className="p-6">
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-gray-800">{getProcessingTitle()}</h2>
            <div className="text-right">
              <div className="text-2xl font-bold text-indigo-600">{progress.overall_percent || 0}%</div>
              <div className="text-xs text-gray-500">{t('processing.overallProgress')}</div>
            </div>
          </div>

          {/* Main Progress Bar */}
          <div className="w-full bg-gray-200 rounded-full h-4 mb-4 overflow-hidden">
            <motion.div
              className="h-4 bg-gradient-to-l from-indigo-500 to-blue-500 rounded-full relative"
              initial={{ width: 0 }}
              animate={{ width: `${progress.overall_percent || 0}%` }}
              transition={{ type: "spring", stiffness: 120, damping: 20 }}
            >
              <motion.div
                className="absolute inset-0 bg-gradient-to-l from-transparent via-white/30 to-transparent"
                animate={{ x: ['-100%', '100%'] }}
                transition={{ repeat: Infinity, duration: 2, ease: 'linear' }}
              />
            </motion.div>
          </div>

          {/* Current Status */}
          <div className="mb-6 flex items-center justify-between text-sm">
            <div className="flex items-center gap-2 text-gray-700">
              <span className="animate-pulse">⚡</span>
              <span>
                {currentStep ? `${t('processing.currentStep')}: ${currentStep.label}` : 
                 completedCount === enhancedSteps.length ? `${t('status.completed')}! ✅` : t('processing.startingProcessing')}
              </span>
            </div>
            {/* Only show an "X of Y steps" counter when there is more than one
                step — "0 of 1" is meaningless for single-step flows (e.g. download-only). */}
            {enhancedSteps.length > 1 && (
              <span className="text-gray-500">
                {completedCount} {t('processing.outOf')} {enhancedSteps.length} {t('processing.stepsCompleted')}
              </span>
            )}
          </div>

          {/* Steps List */}
          <div className="space-y-3">
            {enhancedSteps.map((step, index) => (
              <motion.div
                key={step.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className={`flex items-center gap-4 p-4 rounded-lg border transition-all duration-300 ${
                  step.status === 'completed' 
                    ? 'bg-green-50 border-green-200' 
                    : step.status === 'in_progress'
                    ? 'bg-indigo-50 border-indigo-200'
                    : 'bg-gray-50 border-gray-200'
                }`}
              >
                <div className="text-2xl">{getStepIcon(step.status)}</div>
                
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-2">
                    <span className={`font-medium ${
                      step.status === 'completed' 
                        ? 'text-green-700' 
                        : step.status === 'in_progress'
                        ? 'text-indigo-700'
                        : 'text-gray-600'
                    }`}>
                      {index + 1}. {step.label}
                    </span>
                    <span className="text-xs text-gray-500">
                      {Math.round(step.progress * 100)}%
                    </span>
                  </div>
                  
                  {/* Individual step progress bar */}
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <motion.div
                      className={`h-2 rounded-full ${
                        step.status === 'completed' 
                          ? 'bg-green-500' 
                          : step.status === 'in_progress'
                          ? 'bg-indigo-500'
                          : 'bg-gray-300'
                      }`}
                      initial={{ width: 0 }}
                      animate={{ width: `${step.progress * 100}%` }}
                      transition={{ duration: 0.5 }}
                    />
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Logs Section */}
          {progress.logs && progress.logs.length > 0 && (
            <div className="mt-6 pt-4 border-t border-gray-200">
              <button
                onClick={() => setShowLogs(!showLogs)}
                className="flex items-center gap-2 text-sm font-medium text-gray-600 hover:text-gray-800 transition-colors mb-3"
              >
                <span>{showLogs ? `📋 ${t('processing.hideLogs')}` : `📋 ${t('processing.showLogs')}`}</span>
                <motion.div
                  animate={{ rotate: showLogs ? 180 : 0 }}
                  className="text-xs"
                >
                  {/* eslint-disable-next-line i18next/no-literal-string */}
                  {showLogs ? '▲' : '▼'}
                </motion.div>
              </button>
              
              {showLogs && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="bg-gray-900 text-green-400 rounded-lg p-4 font-mono text-xs max-h-48 overflow-y-auto"
                  dir="ltr"
                >
                  {progress.logs.map((log, index) => (
                    <div key={index} className="mb-1">
                      <span className="text-gray-500 mr-2">{index + 1}:</span>
                      {log}
                    </div>
                  ))}
                </motion.div>
              )}
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
};

export default ProgressDisplay;
