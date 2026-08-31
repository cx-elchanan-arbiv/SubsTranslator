export interface VideoMetadata {
  title: string;
  duration: number;
  duration_string: string;
  view_count: number;
  upload_date: string;
  uploader: string;
  thumbnail: string;
  description: string;
  width: number;
  height: number;
  fps: number;
  filesize: number;
  url: string;
}

export interface FileMetadata {
  filename: string;
  file_size_mb: number;
  size_bytes: number;
  duration: number;
  duration_string: string;
  width: number;
  height: number;
  fps: number;
  mime_type: string;
  extension: string;
  codec_name: string;
  audio_codec: string;
  bit_rate: number;
  thumbnail_url: string | null;
}

export interface UserChoices {
  source_lang: string;
  target_lang: string;
  auto_create_video: boolean;
  whisper_model: WhisperModel;
  url: string;
  translation_service: TranslationService;
  watermark_enabled?: boolean;
  watermark_position?: WatermarkPosition;
  watermark_size?: WatermarkSize;
  subtitle_position?: SubtitlePosition;
}

export type WhisperModel = 'tiny' | 'base' | 'small' | 'medium' | 'large' | 'gemini';
export type TranslationService = 'openai' | 'google';
// Download-only feature. The processing pipeline always needs the video,
// so 'mp3' is offered on the quick-download path alone.
export type DownloadMediaFormat = 'mp4' | 'mp3';

/**
 * "clean" removes spoken filler ("uh", "you know") — subtitles are read, not heard.
 * "faithful" keeps it, for testimony/comedy timing/verbatim reporting.
 */
export type TranslationStyle = 'clean' | 'faithful';

/** Placement of burned-in subtitles inside the video frame. */
export type SubtitlePosition = 'bottom' | 'top' | 'side';

/**
 * The four independent, experimental subtitle-quality toggles.
 *
 * Field names are snake_case because they are sent to the API verbatim and travel
 * unchanged all the way into the Celery task kwargs. All four default to off, in which
 * case the backend runs its legacy pipeline untouched.
 */
export interface SubtitleQualityFlags {
  /** Re-spot cues from Whisper word timestamps instead of its speech segments. */
  spotting_v2: boolean;
  /** Translate with whole-scene context + reading-speed enforcement. */
  translation_v2: boolean;
  /** Filler handling; only meaningful while translation_v2 is on. */
  translation_style: TranslationStyle;
  /** Burn in via a generated .ass file (correct Hebrew shaping) instead of SRT. */
  render_v2: boolean;
}

export const DEFAULT_SUBTITLE_QUALITY_FLAGS: SubtitleQualityFlags = {
  spotting_v2: true,
  translation_v2: true,
  translation_style: 'clean',
  render_v2: true,
};
export type WatermarkPosition = 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
export type WatermarkSize = 'small' | 'medium' | 'large';

export interface ProcessedFiles {
  original_srt?: string;
  translated_srt?: string;
  video_with_subtitles?: string;
}

export interface TimingSummary {
  [key: string]: string;
}

export interface TaskResult {
  title?: string;
  detected_language?: string;
  files?: ProcessedFiles;
  timing_summary?: TimingSummary;
  // How many cues fell back to source text in the translated SRT (bug #2 fix:
  // the job succeeds, but the gap count must be shown, never hidden).
  translation_gaps?: number;
  // The translator that actually RAN (the quality path always uses OpenAI,
  // whatever the selector said) — shown on the results screen for honesty.
  translation_service_used?: string;
  video_metadata?: VideoMetadata;
  user_choices?: UserChoices;
  filename?: string;
  file_size_mb?: number;
  duration?: number;
  download_url?: string;
  message?: string;
  task_id?: string;
  result?: TaskResult;
  status?: TaskStatus;
  error?: string;
}

export type TaskStatus = 'SUCCESS' | 'FAILURE' | 'PENDING' | 'PROGRESS' | 'DOWNLOAD_FAILED';

export type StepStatus = 'waiting' | 'in_progress' | 'completed' | 'error';

export interface Step {
  label: string;
  subtitle: string;
  status: StepStatus;
  progress: number;
  weight: number;
  indeterminate: boolean;
  status_message?: string;
}

export interface Progress {
  steps: Step[];
  overall_percent: number;
  logs: string[];
}

export interface InitialRequest {
  url: string;
  quality?: string;
  type?: string;
}

export interface TaskError {
  code?: string;
  message?: string;
  user_facing_message?: string;
  recoverable?: boolean;
}

export interface TaskStatusResponse {
  task_id: string;
  state: TaskStatus;
  progress: Progress;
  video_metadata?: VideoMetadata | null;
  file_metadata?: FileMetadata | null;
  result?: TaskResult | null;
  user_choices?: UserChoices;
  initial_request?: InitialRequest;
  error?: TaskError | string | null;
}

export interface UploadRequest {
  file: File;
  source_lang: string;
  target_lang: string;
  auto_create_video: boolean;
  whisper_model: WhisperModel;
  translation_service: TranslationService;
  subtitle_position?: SubtitlePosition;
  watermark_enabled?: boolean;
  watermark_position?: WatermarkPosition;
  watermark_size?: WatermarkSize;
  watermark_logo?: File;
}

export interface YoutubeRequest {
  url: string;
  source_lang: string;
  target_lang: string;
  auto_create_video: boolean;
  whisper_model: WhisperModel;
  translation_service: TranslationService;
  subtitle_position?: SubtitlePosition;
  watermark_enabled?: boolean;
  watermark_position?: WatermarkPosition;
  watermark_size?: WatermarkSize;
  watermark_logo?: File;
}

export interface QuickDownloadRequest {
  url: string;
}

export interface TaskInitResponse {
  task_id: string;
  video_metadata?: VideoMetadata;
  file_metadata?: FileMetadata;
  user_choices?: UserChoices;
  initial_request?: InitialRequest;
  progress?: Progress;
}

export interface ApiErrorResponse {
  error: string;
  status: 'error';
  message?: string;
  error_code?: string;
}

export interface Language {
  code: string;
  name: string;
  nativeName: string;
  rtl?: boolean;
}

export interface LanguagesResponse {
  languages: Language[];
}

export interface HealthCheckResponse {
  status: 'healthy' | 'unhealthy';
  timestamp: string;
  services: {
    redis: 'up' | 'down';
    celery: 'up' | 'down';
    backend: 'up' | 'down';
  };
}

export interface DownloadFileParams {
  filename: string;
}

export interface WatermarkConfig {
  enabled: boolean;
  logoFile: File | null;
  logoUrl: string;
  size: WatermarkSize;
  position: WatermarkPosition;
  opacity: number; // 0-100 (percentage)
  isCollapsed?: boolean;
}
