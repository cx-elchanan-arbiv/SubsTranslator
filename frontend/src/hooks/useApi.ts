import { useState, useRef, useCallback, useEffect } from 'react';
import type {
  VideoMetadata,
  FileMetadata,
  UserChoices,
  TaskResult,
  Progress,
  TaskStatusResponse,
  TaskInitResponse,
  WatermarkConfig,
  WhisperModel,
  TranslationService,
  SubtitleQualityFlags,
  SubtitlePosition,
  Step,
  DownloadMediaFormat,
} from '../types';
import { DEFAULT_SUBTITLE_QUALITY_FLAGS } from '../types/api';

export type {
  VideoMetadata,
  FileMetadata,
  UserChoices,
  TaskResult,
  Progress,
  Step,
};

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8081';
const POLLING_INTERVAL = parseInt(process.env.REACT_APP_POLLING_INTERVAL || '3000');

/**
 * Send the four subtitle-quality toggles with a multipart request.
 *
 * They are always sent, including when off: the field names are the same snake_case
 * names the backend task takes, so an explicit "false" documents the request instead of
 * relying on a missing key.
 */
const appendSubtitleFlags = (formData: FormData, flags?: SubtitleQualityFlags) => {
  const resolved = flags || DEFAULT_SUBTITLE_QUALITY_FLAGS;
  formData.append('spotting_v2', String(resolved.spotting_v2));
  formData.append('translation_v2', String(resolved.translation_v2));
  formData.append('translation_style', resolved.translation_style);
  formData.append('render_v2', String(resolved.render_v2));
};

export const useApi = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState<Progress>({
    steps: [],
    overall_percent: 0,
    logs: []
  });
  const [result, setResult] = useState<TaskResult | null>(null);
  const [salvagedResult, setSalvagedResult] = useState<{ files?: Record<string, string> } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [videoMetadata, setVideoMetadata] = useState<VideoMetadata | undefined>(undefined);
  const [fileMetadata, setFileMetadata] = useState<FileMetadata | undefined>(undefined);
  const [userChoices, setUserChoices] = useState<UserChoices | undefined>(undefined);
  const [initialRequest, setInitialRequest] = useState<any>(undefined);
  const [currentProcessingType, setCurrentProcessingType] = useState<'file_upload' | 'youtube_full' | 'youtube_download' | null>(null);
  const pollTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // ===== P0 Guards & Refs for stability =====
  const isProcessingRef = useRef(isProcessing);
  const resultRef = useRef(result);
  const videoMetadataRef = useRef(videoMetadata);
  const fileMetadataRef = useRef(fileMetadata);
  const userChoicesRef = useRef(userChoices);
  const initialRequestRef = useRef(initialRequest);

  // Flow control: increment this to invalidate old async operations
  const flowRef = useRef(0);

  // Dedupe: track last result ID to avoid redundant updates
  const lastSetIdRef = useRef<string | null>(null);

  // Canonical task ID from server (handles task chains/redirects)
  const activeTaskIdRef = useRef<string | null>(null);

  // Keep refs in sync with state
  useEffect(() => { isProcessingRef.current = isProcessing; }, [isProcessing]);
  useEffect(() => { resultRef.current = result; }, [result]);
  useEffect(() => { videoMetadataRef.current = videoMetadata; }, [videoMetadata]);
  useEffect(() => { fileMetadataRef.current = fileMetadata; }, [fileMetadata]);
  useEffect(() => { userChoicesRef.current = userChoices; }, [userChoices]);
  useEffect(() => { initialRequestRef.current = initialRequest; }, [initialRequest]);

  // Safe setResult with dedupe to prevent redundant updates
  const safeSetResult = useCallback((enrichedResult: TaskResult | null) => {
    if (!enrichedResult) {
      setResult(null);
      lastSetIdRef.current = null;
      return;
    }

    const taskId = enrichedResult.task_id;
    if (!taskId) {
      return;
    }

    // Dedupe: if we already set this exact result, skip
    if (lastSetIdRef.current === taskId && resultRef.current?.task_id === taskId) {
      return;
    }

    setResult(enrichedResult);
    lastSetIdRef.current = taskId;
  }, []);

  // BroadcastChannel for multi-tab coordination
  const channelRef = useRef<BroadcastChannel | null>(null);

  // Initialize BroadcastChannel
  useEffect(() => {
    if (typeof BroadcastChannel !== 'undefined') {
      channelRef.current = new BroadcastChannel('task_processing');

      channelRef.current.onmessage = (event) => {
        if (event.data.type === 'TASK_STARTED') {
          // Another tab started processing this task
          if (event.data.taskId && pollTimeoutRef.current) {
            clearTimeout(pollTimeoutRef.current);
            pollTimeoutRef.current = null;
          }
        }
      };
    }

    return () => {
      channelRef.current?.close();
    };
  }, []);

  // Validate task exists before resuming
  const validateTask = async (taskId: string): Promise<{valid: boolean; data?: any}> => {
    try {
      const response = await fetch(`${API_BASE_URL}/status/${taskId}`, {
        credentials: 'include'
      });
      if (!response.ok) {
        return { valid: false };
      }
      const data = await response.json();
      // Check that we got valid task data with a state
      if (data && data.state) {
        return { valid: true, data };
      }
      return { valid: false };
    } catch {
      return { valid: false };
    }
  };

  // Resume task from URL or localStorage
  useEffect(() => {
    const resumeTask = async () => {
      // Priority: URL params > localStorage
      const urlParams = new URLSearchParams(window.location.search);
      const urlTaskId = urlParams.get('task');
      const savedTaskId = localStorage.getItem('active_task_id');

      const taskToResume = urlTaskId || savedTaskId;

      if (taskToResume && !isProcessing && !result && !error) {
        // Mark this as a new flow to invalidate any previous operations
        const myFlow = ++flowRef.current;

        // Validate task exists before resuming
        const validation = await validateTask(taskToResume);

        // Check if this flow is still valid (not cancelled by a new operation)
        if (myFlow !== flowRef.current) {
          return;
        }

        if (validation.valid && validation.data) {
          const data = validation.data;

          // Extract canonical task_id from server response
          const canonicalTaskId = data.result?.task_id || taskToResume;
          activeTaskIdRef.current = canonicalTaskId;

          // Restore metadata if available
          if (data.video_metadata) {
            setVideoMetadata(data.video_metadata);
          }
          if (data.file_metadata) {
            setFileMetadata(data.file_metadata);
          }
          if (data.user_choices && Object.keys(data.user_choices).length > 0) {
            setUserChoices(data.user_choices);
          }
          if (data.initial_request && Object.keys(data.initial_request).length > 0) {
            setInitialRequest(data.initial_request);
          }

          // Sync URL and localStorage with canonical ID
          if (!urlTaskId || urlTaskId !== canonicalTaskId) {
            const newUrl = new URL(window.location.href);
            newUrl.searchParams.set('task', canonicalTaskId);
            window.history.replaceState({}, '', newUrl.toString());
          }
          if (!savedTaskId || savedTaskId !== canonicalTaskId) {
            localStorage.setItem('active_task_id', canonicalTaskId);
          }

          // Handle based on state
          if (data.state === 'SUCCESS') {
            // Task already completed - restore result without polling
            const currentResult = data.result?.result || data.result;

            if (!currentResult) {
              // Stale task: completed long ago and its result payload is gone
              // (e.g. output files cleaned up). Resuming it would strand the UI
              // on a finished-but-empty progress card — treat it like an
              // invalid task and return to the home screen instead.
              activeTaskIdRef.current = null;
              localStorage.removeItem('active_task_id');
              const staleUrl = new URL(window.location.href);
              staleUrl.searchParams.delete('task');
              window.history.replaceState({}, '', staleUrl.toString());
              return;
            }

            const enrichedResult = {
              ...currentResult,
              task_id: canonicalTaskId,
              user_choices: data.user_choices || currentResult.user_choices
            };

            safeSetResult(enrichedResult);
            setIsProcessing(false);
            setProgress(prev => ({
              ...prev,
              overall_percent: 100,
              steps: prev.steps.map(step => ({ ...step, status: 'completed', progress: 100 }))
            }));
          } else if (data.state === 'PROGRESS' || data.state === 'PENDING') {
            // Task still in progress - resume polling
            if (data.progress) {
              setProgress(data.progress);
            }
            setIsProcessing(true);
            pollStatus(canonicalTaskId, false);
          } else if (data.state === 'FAILURE') {
            // Task failed - show error
            setIsProcessing(false);
            setError(data.error?.user_facing_message || data.error?.message || 'העיבוד נכשל');
          }
        } else {
          localStorage.removeItem('active_task_id');
          const newUrl = new URL(window.location.href);
          newUrl.searchParams.delete('task');
          window.history.replaceState({}, '', newUrl.toString());
        }
      }
    };

    resumeTask();
  }, []); // Only on mount

  const pollStatus = useCallback((taskId: string, isQuickDownload = false) => {
    if (pollTimeoutRef.current) {
      clearTimeout(pollTimeoutRef.current);
    }

    // Early guard: Skip stale polls (we've moved on to a different task)
    if (activeTaskIdRef.current && taskId !== activeTaskIdRef.current) {
      return;
    }

    // Capture current flow to detect if we're invalidated
    const myFlow = flowRef.current;

    const poll = async () => {
      try {
        // Check if this polling is still valid
        if (myFlow !== flowRef.current) {
          return;
        }

        const response = await fetch(`${API_BASE_URL}/status/${taskId}`, {
          credentials: 'include'
        });
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }

        const data: TaskStatusResponse = await response.json();

        // Check again after async operation
        if (myFlow !== flowRef.current) {
          return;
        }

        // Handle new unified schema - metadata comes directly from the response
        // Use refs to check current values instead of stale closure values
        if (data.video_metadata && !videoMetadataRef.current) {
          setVideoMetadata(data.video_metadata);
        }
        if (data.file_metadata && !fileMetadataRef.current) {
          setFileMetadata(data.file_metadata);
        }
        if (data.user_choices && Object.keys(data.user_choices).length > 0 && !userChoicesRef.current) {
          setUserChoices(data.user_choices);
        }
        if (data.initial_request && Object.keys(data.initial_request).length > 0 && !initialRequestRef.current) {
          setInitialRequest(data.initial_request);
        }

        const currentResult = data.result?.result || data.result;

        if (data.state === 'PROGRESS') {
          // Update progress from unified schema
          const progressData = data.progress || { steps: [], overall_percent: 0, logs: [] };
          setProgress(progressData);
        } else if (data.state === 'SUCCESS') {
          if (currentResult?.status === 'FAILURE') {
            if (pollTimeoutRef.current) {
              clearTimeout(pollTimeoutRef.current);
              pollTimeoutRef.current = null;
            }
            setIsProcessing(false);
            setError(currentResult.error || 'העיבוד נכשל');
            setProgress(prev => ({
              ...prev,
              steps: prev.steps.map(step => step.status === 'in_progress' ? { ...step, status: 'error' } : step)
            }));

            // Clean up localStorage and URL on failure
            localStorage.removeItem('active_task_id');
            const newUrl = new URL(window.location.href);
            newUrl.searchParams.delete('task');
            window.history.replaceState({}, '', newUrl.toString());

            return;
          }
          
          // Handle download-specific statuses
          if (currentResult?.status === 'DOWNLOAD_FAILED') {
            if (pollTimeoutRef.current) {
              clearTimeout(pollTimeoutRef.current);
              pollTimeoutRef.current = null;
            }
            setIsProcessing(false);
            isProcessingRef.current = false;  // Stop polling immediately to prevent race condition
            setCurrentProcessingType(null);
            setError(currentResult.error || 'ההורדה נכשלה');
            setProgress(prev => ({
              ...prev,
              steps: prev.steps.map(step => step.status === 'in_progress' ? { ...step, status: 'error' } : step)
            }));

            // Clean up localStorage and URL on download failure
            localStorage.removeItem('active_task_id');
            const newUrl = new URL(window.location.href);
            newUrl.searchParams.delete('task');
            window.history.replaceState({}, '', newUrl.toString());

            return;
          }
          
          const nextTaskId = data.result?.task_id;

          if (nextTaskId && nextTaskId !== taskId) {
            // Check if we already started polling this task (prevent duplicate polls)
            if (activeTaskIdRef.current === nextTaskId) {
              return;
            }

            // Server returned a different task_id - this is a chain, follow it
            activeTaskIdRef.current = nextTaskId;

            // Invalidate all old polls on the previous task
            flowRef.current++;

            // Update URL and localStorage with canonical ID
            try {
              localStorage.setItem('active_task_id', nextTaskId);
              const newUrl = new URL(window.location.href);
              newUrl.searchParams.set('task', nextTaskId);
              window.history.replaceState({}, '', newUrl.toString());
            } catch (e) {
              // Silently fail - not critical for functionality
            }

            pollStatus(nextTaskId, isQuickDownload);
            return; // Critical: stop this poll loop, new one started
          } else {
            // Task completed successfully
            const canonicalTaskId = nextTaskId || taskId;
            activeTaskIdRef.current = canonicalTaskId;

            setIsProcessing(false);
            isProcessingRef.current = false;  // Stop polling immediately to prevent race condition
            setCurrentProcessingType(null);
            setProgress(prev => ({
              ...prev,
              overall_percent: 100,
              steps: prev.steps.map(step => ({ ...step, status: 'completed', progress: 100 }))
            }));

            // Include task_id and user_choices in result for summary feature
            const enrichedResult = currentResult ? {
              ...currentResult,
              task_id: canonicalTaskId,
              user_choices: data.user_choices || currentResult.user_choices
            } : null;

            // Use safeSetResult to avoid duplicate updates
            safeSetResult(enrichedResult);

            // ✅ P0.5: Don't clean up on SUCCESS - keep task in URL/localStorage for refresh recovery
            // Cleanup will happen in resetState() or when starting a new task
          }
        } else if (data.state === 'FAILURE') {
          if (pollTimeoutRef.current) {
            clearTimeout(pollTimeoutRef.current);
            pollTimeoutRef.current = null;
          }
          setIsProcessing(false);
          isProcessingRef.current = false;  // Stop polling immediately to prevent race condition
          setCurrentProcessingType(null);

          // Handle new error structure (can be object or string)
          let errorMessage = 'העיבוד נכשל';
          let originalError = '';
          if (typeof data.error === 'string') {
            errorMessage = data.error;
            originalError = data.error;
          } else if (data.error && typeof data.error === 'object') {
            errorMessage = data.error.user_facing_message ||
                          data.error.message ||
                          'העיבוד נכשל';
            originalError = data.error.message || '';
          }

          // Extract specific error details (e.g., "Private video")
          const errorDetails = originalError.match(/ERROR: \[youtube\] [^:]+: (.+?)(?:\. |$)/);
          if (errorDetails && errorDetails[1]) {
            errorMessage = `${errorMessage}\n\n${errorDetails[1]}`;
          } else if (originalError && originalError !== errorMessage) {
            // If we have original error that's different, append it
            errorMessage = `${errorMessage}\n\n${originalError}`;
          }

          // Pass the full error object to preserve the code field for bot detection UI
          setError(data.error || errorMessage);

          // A failed run can still have produced files worth keeping — the transcript
          // survives a failed translation, the .srt survives a failed burn — and the
          // task marks that payload `salvaged`. Holding on to it is what lets the
          // error screen offer a download instead of leaving the user with nothing
          // after a transcription that already cost time and money.
          setSalvagedResult(
            data.result && (data.result as any).salvaged ? (data.result as any) : null
          );

          setProgress(prev => ({
            ...prev,
            steps: prev.steps.map(step => step.status === 'in_progress' ? { ...step, status: 'error' } : step)
          }));

          // Clean up localStorage and URL on failure
          localStorage.removeItem('active_task_id');
          const newUrl = new URL(window.location.href);
          newUrl.searchParams.delete('task');
          window.history.replaceState({}, '', newUrl.toString());
        }
      } catch (err) {
        if (pollTimeoutRef.current) {
          clearTimeout(pollTimeoutRef.current);
          pollTimeoutRef.current = null;
        }
        setIsProcessing(false);
        isProcessingRef.current = false;  // Stop polling immediately to prevent race condition
        setCurrentProcessingType(null);
        setError('שגיאה בקבלת סטטוס המשימה');

        // Clean up localStorage and URL on error
        localStorage.removeItem('active_task_id');
        const newUrl = new URL(window.location.href);
        newUrl.searchParams.delete('task');
        window.history.replaceState({}, '', newUrl.toString());
      }

      if (isProcessingRef.current) {
        pollTimeoutRef.current = setTimeout(poll, POLLING_INTERVAL);
      }
    };

    pollTimeoutRef.current = setTimeout(poll, 100);
  }, [safeSetResult]); // Only depends on safeSetResult, all other values accessed via refs


  const startProcessing = useCallback(async (endpoint: string, options: RequestInit, processingType: 'file_upload' | 'youtube_full' | 'youtube_download' = 'file_upload') => {
    // Start a new flow - this invalidates any previous async operations
    flowRef.current++;
    lastSetIdRef.current = null;
    activeTaskIdRef.current = null;

    setIsProcessing(true);
    setProgress({
      steps: [],
      overall_percent: 0,
      logs: []
    });
    safeSetResult(null);
    setError(null);
    setSalvagedResult(null);
    setVideoMetadata(undefined);
    setFileMetadata(undefined);
    setUserChoices(undefined);
    setInitialRequest(undefined);
    setCurrentProcessingType(processingType);

    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        ...options,
        credentials: 'include'
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Request failed');
      }

      const responseData = await response.json();
      const { task_id, video_metadata, file_metadata, user_choices, initial_request, progress } = responseData;

      // Handle immediate data from 202 response
      if (video_metadata) {
        setVideoMetadata(video_metadata);
      }
      if (file_metadata) {
        setFileMetadata(file_metadata);
      }
      if (user_choices && Object.keys(user_choices).length > 0) {
        setUserChoices(user_choices);
      }
      if (initial_request && Object.keys(initial_request).length > 0) {
        setInitialRequest(initial_request);
      }
      if (progress) {
        setProgress(progress);
      }

      if (task_id) {
        // Save task_id as active canonical ID
        activeTaskIdRef.current = task_id;

        // Save task_id to localStorage and URL
        localStorage.setItem('active_task_id', task_id);
        const newUrl = new URL(window.location.href);
        newUrl.searchParams.set('task', task_id);
        window.history.replaceState({}, '', newUrl.toString());

        // Broadcast to other tabs
        if (channelRef.current) {
          channelRef.current.postMessage({
            type: 'TASK_STARTED',
            taskId: task_id
          });
        }

        pollStatus(task_id, false);
      } else {
        throw new Error('Did not receive a task ID');
      }
    } catch (err: any) {
      setError(err.message || 'שגיאה לא צפויה');
      setIsProcessing(false);
      setCurrentProcessingType(null);
    }
  }, [pollStatus]);

  const onFileUpload = async (
    file: File,
    sourceLang: string,
    targetLang: string,
    autoCreateVideo: boolean,
    whisperModel: WhisperModel,
    translationService: TranslationService,
    watermarkConfig?: WatermarkConfig,
    subtitleFlags?: SubtitleQualityFlags,
    subtitlePosition: SubtitlePosition = 'bottom'
  ) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('source_lang', sourceLang);
    formData.append('target_lang', targetLang);
    formData.append('auto_create_video', String(autoCreateVideo));
    formData.append('whisper_model', whisperModel);
    formData.append('translation_service', translationService);
    appendSubtitleFlags(formData, subtitleFlags);
    formData.append('subtitle_position', subtitlePosition);

    // Add watermark configuration
    if (watermarkConfig) {
      formData.append('watermark_enabled', String(watermarkConfig.enabled));
      if (watermarkConfig.enabled) {
        formData.append('watermark_position', watermarkConfig.position || 'top-right');
        formData.append('watermark_size', watermarkConfig.size || 'medium');
        formData.append('watermark_opacity', String(watermarkConfig.opacity || 40));

        // Handle logo - either from file or from saved data URL
        if (watermarkConfig.logoFile) {
          formData.append('watermark_logo', watermarkConfig.logoFile);
        } else if (watermarkConfig.logoUrl) {
          // Convert data URL back to File object for consistency
          try {
            const response = await fetch(watermarkConfig.logoUrl);
            const blob = await response.blob();
            const file = new File([blob], 'custom_logo.png', { type: blob.type });
            formData.append('watermark_logo', file);
          } catch (error) {
            // Silently fail - logo conversion not critical
          }
        }
      }
    } else {
      formData.append('watermark_enabled', 'false');
    }

    startProcessing('/upload', {
      method: 'POST',
      body: formData,
    }, 'file_upload');
  };

  const onYoutubeSubmit = async (
    youtubeUrl: string,
    sourceLang: string,
    targetLang: string,
    autoCreateVideo: boolean,
    whisperModel: WhisperModel,
    translationService: TranslationService,
    watermarkConfig?: WatermarkConfig,
    startTime?: string,
    endTime?: string,
    subtitleFlags?: SubtitleQualityFlags,
    subtitlePosition: SubtitlePosition = 'bottom'
  ) => {
    // Check if we have a custom logo (either file or saved URL) - if so, use FormData
    const hasCustomLogo = watermarkConfig?.enabled && (watermarkConfig?.logoFile || watermarkConfig?.logoUrl);
    
    if (hasCustomLogo) {
      // Use FormData when we have a custom logo file
      const formData = new FormData();
      formData.append('url', youtubeUrl);
      formData.append('source_lang', sourceLang);
      formData.append('target_lang', targetLang);
      formData.append('auto_create_video', String(autoCreateVideo));
      formData.append('whisper_model', whisperModel);
      formData.append('translation_service', translationService);
      appendSubtitleFlags(formData, subtitleFlags);
      formData.append('subtitle_position', subtitlePosition);

      // Add time range if provided
      if (startTime && endTime) {
        formData.append('start_time', startTime);
        formData.append('end_time', endTime);
      }

      // Add watermark configuration
      formData.append('watermark_enabled', String(watermarkConfig.enabled));
      if (watermarkConfig.enabled) {
        formData.append('watermark_position', watermarkConfig.position || 'top-right');
        formData.append('watermark_size', watermarkConfig.size || 'medium');
        formData.append('watermark_opacity', String(watermarkConfig.opacity || 40));

        // Handle logo - either from file or from saved data URL
        if (watermarkConfig.logoFile) {
          formData.append('watermark_logo', watermarkConfig.logoFile);
        } else if (watermarkConfig.logoUrl) {
          // Convert data URL back to File object for consistency
          try {
            const response = await fetch(watermarkConfig.logoUrl);
            const blob = await response.blob();
            const file = new File([blob], 'custom_logo.png', { type: blob.type });
            formData.append('watermark_logo', file);
          } catch (error) {
            // Silently fail - logo conversion not critical
          }
        }
      }
      
      startProcessing('/youtube', {
        method: 'POST',
        body: formData,
      }, 'youtube_full');
    } else {
      // Use JSON when no custom logo
      const requestBody: any = {
        url: youtubeUrl,
        source_lang: sourceLang,
        target_lang: targetLang,
        auto_create_video: autoCreateVideo,
        whisper_model: whisperModel,
        translation_service: translationService,
        ...(subtitleFlags || DEFAULT_SUBTITLE_QUALITY_FLAGS),
        subtitle_position: subtitlePosition,
      };

      // Add time range if provided
      if (startTime && endTime) {
        requestBody.start_time = startTime;
        requestBody.end_time = endTime;
      }

      // Add watermark configuration
      if (watermarkConfig) {
        requestBody.watermark_enabled = watermarkConfig.enabled;
        if (watermarkConfig.enabled) {
          requestBody.watermark_position = watermarkConfig.position || 'top-right';
          requestBody.watermark_size = watermarkConfig.size || 'medium';
          requestBody.watermark_opacity = watermarkConfig.opacity || 40;
        }
      } else {
        requestBody.watermark_enabled = false;
      }
      
      startProcessing('/youtube', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      }, 'youtube_full');
    }
  };

  const handleQuickDownload = (
    youtubeUrl: string,
    startTime?: string,
    endTime?: string,
    mediaFormat: DownloadMediaFormat = 'mp4',
  ) => {
    const requestBody: {url: string; start_time?: string; end_time?: string; media_format: DownloadMediaFormat} = {
      url: youtubeUrl,
      media_format: mediaFormat,
    };

    // Add time range if provided
    if (startTime && endTime) {
      requestBody.start_time = startTime;
      requestBody.end_time = endTime;
    }

    startProcessing('/download-video-only', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
    }, 'youtube_download');
  };

  const resetState = useCallback((options?: { reason?: string; force?: boolean }) => {
    const reason = options?.reason || 'unspecified';
    const force = options?.force || false;

    // Guard: if we have an active result that matches activeTaskId, don't clear (unless forced)
    const currentResult = resultRef.current;
    const activeId = activeTaskIdRef.current;

    if (!force && currentResult?.task_id && currentResult.task_id === activeId) {
      return;
    }

    // Invalidate any in-flight async operations
    flowRef.current++;

    setIsProcessing(false);
    setProgress({
      steps: [],
      overall_percent: 0,
      logs: []
    });
    safeSetResult(null);
    setError(null);
    setSalvagedResult(null);
    setVideoMetadata(undefined);
    setFileMetadata(undefined);
    setUserChoices(undefined);
    setInitialRequest(undefined);
    setCurrentProcessingType(null);

    // Clear refs
    lastSetIdRef.current = null;
    activeTaskIdRef.current = null;

    // Clear any ongoing polling
    if (pollTimeoutRef.current) {
      clearTimeout(pollTimeoutRef.current);
      pollTimeoutRef.current = null;
    }

    // Clean up localStorage and URL
    localStorage.removeItem('active_task_id');
    const newUrl = new URL(window.location.href);
    newUrl.searchParams.delete('task');
    window.history.replaceState({}, '', newUrl.toString());
  }, [safeSetResult]);
  
  return {
    isProcessing,
    progress,
    result,
    error,
    salvagedResult,
    videoMetadata,
    fileMetadata,
    userChoices,
    initialRequest,
    currentProcessingType,
    onFileUpload,
    onYoutubeSubmit,
    handleQuickDownload,
    resetState
  };
};
