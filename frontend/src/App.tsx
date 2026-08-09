import { useState, useEffect } from 'react';
import './App.css';
import { useApi } from './hooks/useApi';
import { useI18n } from './i18n/I18nProvider';
import { useTranslation } from 'react-i18next';
import { useAuth } from './contexts/AuthContext';
import type {
  WhisperModel,
  TranslationService,
  WatermarkConfig,
  SubtitleQualityFlags,
  SubtitlePosition,
} from './types';
import { DEFAULT_SUBTITLE_QUALITY_FLAGS } from './types/api';
import LanguageSelection from './components/LanguageSelection';
import LanguageSelector from './components/LanguageSelector';
import Options from './components/Options';
import Tabs from './components/Tabs';
import UploadForm from './components/UploadForm';
import YoutubeForm from './components/YoutubeForm';
import ProgressDisplay from './components/ProgressDisplay';
import ResultsDisplay from './components/ResultsDisplay';
import AuthModal from './components/AuthModal';
import WatermarkSettings from './components/WatermarkSettings';
import LogoOnlyForm from './components/LogoOnlyForm';
import VideoCutterForm from './components/VideoCutterForm';
import EmbedSubtitlesForm from './components/EmbedSubtitlesForm';
import VideoMergerForm from './components/VideoMergerForm';
import AudioExtractorForm from './components/AudioExtractorForm';

import Header from './components/Header';
import HeroSection from './components/HeroSection';


interface Language {
  [key: string]: string;
}

function App() {
  const { direction, language } = useI18n();
  const { t } = useTranslation(['common', 'errors']);
  const { loading } = useAuth();
  const [activeTab, setActiveTab] = useState<'upload' | 'youtube'>('youtube');
  const [sourceLang, setSourceLang] = useState('auto');
  const [targetLang, setTargetLang] = useState('he');
  const [languages, setLanguages] = useState<Language>({});
  const [autoCreateVideo, setAutoCreateVideo] = useState(true);
  const [whisperModel, setWhisperModel] = useState<WhisperModel>('base');
  const [translationService, setTranslationService] = useState<TranslationService>('openai');
  const [transcriptionOnly, setTranscriptionOnly] = useState(false);
  // Experimental subtitle-quality toggles — all off by default (= legacy pipeline).
  const [subtitleFlags, setSubtitleFlags] = useState<SubtitleQualityFlags>(DEFAULT_SUBTITLE_QUALITY_FLAGS);
  const [subtitlePosition, setSubtitlePosition] = useState<SubtitlePosition>('bottom');
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [selectedAdvancedFeature, setSelectedAdvancedFeature] = useState<string | null>(null);
  const [watermarkConfig, setWatermarkConfig] = useState<WatermarkConfig>({
    enabled: false,
    logoFile: null,
    logoUrl: '',
    size: 'medium',
    position: 'bottom-right',
    opacity: 40, // 40% default opacity
    isCollapsed: false
  });

  // Local state for logo-only feature
  const [isLogoProcessing, setIsLogoProcessing] = useState(false);
  const [logoError, setLogoError] = useState<string | null>(null);

  // Selected file for upload (two-step flow: select file, then start processing)
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // Feature flags
  const [youtubeDownloadEnabled, setYoutubeDownloadEnabled] = useState(true); // Default true
  const [youtubeRestricted, setYoutubeRestricted] = useState(false); // PRO-only in hosted mode

  const {
    isProcessing,
    progress,
    result,
    error,
    videoMetadata,
    fileMetadata,
    userChoices,
    initialRequest,
    currentProcessingType,
    onFileUpload,
    onYoutubeSubmit,
    handleQuickDownload,
    resetState,
  } = useApi();

  useEffect(() => {
    const apiBaseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8081';

    // Load languages from backend API with fallback to local translations
    fetch(`${apiBaseUrl}/languages`, {
      headers: {
        'Accept-Language': String(language)
      },
      credentials: 'include'
    })
      .then(res => res.json())
      .then(data => {
        if (data && Object.keys(data).length > 0) {
          setLanguages(data);
        } else {
          // Fallback to local translations if API fails
          throw new Error('Empty response from API');
        }
      })
      .catch(() => {
        // Fallback: use local translations with shared config
        const sharedConfig = require('./shared-config.js');
        const languageMap: Language = {};

        // Include auto-detect option
        languageMap['auto'] = t(`common:languages.auto`, { defaultValue: 'Auto Detect' });

        // Add all available languages from shared config
        Object.keys(sharedConfig.TRANSCRIPTION_LANGUAGES).forEach(key => {
          languageMap[key] = t(`common:languages.${key}`, {
            defaultValue: sharedConfig.getLanguageName(key)
          });
        });

        setLanguages(languageMap);
      });

    // Load feature flags (only once, not dependent on activeTab)
    fetch(`${apiBaseUrl}/api/features`, {
      credentials: 'include'
    })
      .then(res => res.json())
      .then(data => {
        if (data.youtube_download_enabled !== undefined) {
          setYoutubeDownloadEnabled(data.youtube_download_enabled);
        }
        if (data.youtube_restricted !== undefined) {
          setYoutubeRestricted(data.youtube_restricted);
        }
      })
      .catch(() => {
        // Keep defaults on error
      });
  }, [language, t]);

  // Separate effect: Handle activeTab when YouTube is disabled or restricted
  useEffect(() => {
    if ((!youtubeDownloadEnabled || youtubeRestricted) && activeTab === 'youtube') {
      setActiveTab('upload');
    }
  }, [youtubeDownloadEnabled, youtubeRestricted, activeTab]);

  // Switch from Gemini when user goes to upload tab (Gemini only works with YouTube)
  useEffect(() => {
    if (activeTab === 'upload' && whisperModel === 'gemini') {
      setWhisperModel('base');
    }
  }, [activeTab, whisperModel]);
  
  if (loading) {
    return (
      <div className="app min-h-screen bg-dark-bg text-white" dir={direction}>
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-center justify-center min-h-screen">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
              <div>{t('common:status.loading')}</div>
            </div>
          </div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="app min-h-screen bg-dark-bg text-white" dir={direction}>

      
      <Header onShowAuthModal={() => setShowAuthModal(true)} onHomeClick={() => {
        resetState({ reason: 'home-button-click', force: true });
        setSelectedFile(null);
      }} />
      
      {/* Language Selector - Fixed top-right, next to logo */}
      <div className="fixed top-3 right-16 z-50">
        <LanguageSelector compact />
      </div>
      
      {!isProcessing && !result && (
        <HeroSection />
      )}
      
      <div className="container mx-auto px-4 py-8">

        {(isProcessing || result || error) && (
          <>
            <Tabs
              activeTab={activeTab}
              onTabChange={setActiveTab}
              disabled={isProcessing}
              youtubeEnabled={youtubeDownloadEnabled}
              youtubeRestricted={youtubeRestricted}
            />
            
            <main className="main-content">
          {isProcessing || error ? (
            <ProgressDisplay
              isProcessing={isProcessing}
              progress={progress}
              error={error || ''}
              videoMetadata={videoMetadata}
              fileMetadata={fileMetadata}
              userChoices={userChoices}
              initialRequest={initialRequest}
              processingType={currentProcessingType}
              onRetry={() => {
                resetState({ reason: 'retry-after-error', force: true });
                setSelectedFile(null);
              }}
            />
          ) : result ? (
            <ResultsDisplay
              result={result}
              autoCreateVideo={autoCreateVideo}
              onStartNew={() => {
                resetState({ reason: 'start-new-task', force: true });
                setSelectedFile(null);
              }}
            />
          ) : null}
            </main>
          </>
        )}
        
        {!isProcessing && !result && !error && (
          <>
            <Tabs
              activeTab={activeTab}
              onTabChange={setActiveTab}
              disabled={isProcessing}
              youtubeEnabled={youtubeDownloadEnabled}
              youtubeRestricted={youtubeRestricted}
            />
            
            <main className="main-content">
              <LanguageSelection
                sourceLang={sourceLang}
                targetLang={targetLang}
                languages={languages}
                onSourceLangChange={setSourceLang}
                onTargetLangChange={setTargetLang}
                disabled={isProcessing}
              />
              <Options
                autoCreateVideo={autoCreateVideo}
                onAutoCreateVideoChange={setAutoCreateVideo}
                whisperModel={whisperModel}
                onWhisperModelChange={setWhisperModel}
                translationService={translationService}
                onTranslationServiceChange={setTranslationService}
                transcriptionOnly={transcriptionOnly}
                onTranscriptionOnlyChange={setTranscriptionOnly}
                subtitleFlags={subtitleFlags}
                onSubtitleFlagsChange={setSubtitleFlags}
                subtitlePosition={subtitlePosition}
                onSubtitlePositionChange={setSubtitlePosition}
                disabled={isProcessing}
                activeTab={activeTab}
              />
              <WatermarkSettings
                config={watermarkConfig}
                onChange={setWatermarkConfig}
                disabled={isProcessing}
              />

              {activeTab === 'upload' && (
                <>
                  <UploadForm
                    selectedFile={selectedFile}
                    onFileSelect={setSelectedFile}
                    isProcessing={isProcessing}
                  />
                  {selectedFile && !isProcessing && (
                    <div className="start-processing-container" style={{ marginTop: '1rem', textAlign: 'center' }}>
                      <button
                        className="btn-primary"
                        onClick={() => {
                          onFileUpload(
                            selectedFile,
                            sourceLang,
                            transcriptionOnly ? '' : targetLang,
                            autoCreateVideo,
                            whisperModel,
                            translationService,
                            watermarkConfig,
                            subtitleFlags,
                            subtitlePosition,
                          );
                          setSelectedFile(null);
                        }}
                        disabled={isProcessing}
                      >
                        🚀 {t('common:startProcessing')}
                      </button>
                    </div>
                  )}
                </>
              )}
              {activeTab === 'youtube' && (
                <YoutubeForm
                  onYoutubeSubmit={(url) => onYoutubeSubmit(
                    url,
                    sourceLang,
                    transcriptionOnly ? '' : targetLang,
                    autoCreateVideo,
                    whisperModel,
                    translationService,
                    watermarkConfig,
                    undefined,
                    undefined,
                    subtitleFlags,
                    subtitlePosition,
                  )}
                  onQuickDownload={handleQuickDownload}
                  isProcessing={isProcessing}
                  sourceLang={sourceLang}
                  targetLang={transcriptionOnly ? '' : targetLang}
                  autoCreateVideo={autoCreateVideo}
                  whisperModel={whisperModel}
                  translationService={translationService}
                />
              )}

              {/* Advanced Features - After Upload Area */}
              <div className="advanced-features-section">
                <div className="advanced-features-buttons">
                  <button
                    className={`advanced-feature-btn ${selectedAdvancedFeature === 'logo-only' ? 'active' : ''}`}
                    onClick={() => setSelectedAdvancedFeature(selectedAdvancedFeature === 'logo-only' ? null : 'logo-only')}
                    disabled={isProcessing}
                  >
                    🎨 {t('advanced.logoOnly')}
                  </button>
                  <button
                    className={`advanced-feature-btn ${selectedAdvancedFeature === 'video-cutter' ? 'active' : ''}`}
                    onClick={() => setSelectedAdvancedFeature(selectedAdvancedFeature === 'video-cutter' ? null : 'video-cutter')}
                    disabled={isProcessing}
                  >
                    ✂️ {t('advanced.videoCutter')}
                  </button>
                  <button
                    className={`advanced-feature-btn ${selectedAdvancedFeature === 'embed-subtitles' ? 'active' : ''}`}
                    onClick={() => setSelectedAdvancedFeature(selectedAdvancedFeature === 'embed-subtitles' ? null : 'embed-subtitles')}
                    disabled={isProcessing}
                  >
                    📝 {t('advanced.embedSubtitles')}
                  </button>
                  <button
                    className={`advanced-feature-btn ${selectedAdvancedFeature === 'video-merger' ? 'active' : ''}`}
                    onClick={() => setSelectedAdvancedFeature(selectedAdvancedFeature === 'video-merger' ? null : 'video-merger')}
                    disabled={isProcessing}
                  >
                    🔗 {t('advanced.videoMerger')}
                  </button>
                  <button
                    className={`advanced-feature-btn ${selectedAdvancedFeature === 'audio-extractor' ? 'active' : ''}`}
                    onClick={() => setSelectedAdvancedFeature(selectedAdvancedFeature === 'audio-extractor' ? null : 'audio-extractor')}
                    disabled={isProcessing}
                  >
                    🎵 {t('advanced.audioExtractor')}
                  </button>
                </div>

                {/* Advanced Feature Content */}
                {selectedAdvancedFeature === 'logo-only' && (
                  <div className="advanced-feature-content">
                    <h3>🎨 {t('advanced.logoOnly')}</h3>
                    <p className="feature-description">{t('advanced.logoOnlyDescription')}</p>

                    {logoError && (
                      <div className="error-message" style={{ marginBottom: '1rem', padding: '1rem', background: '#fee', border: '1px solid #fcc', borderRadius: '8px', color: '#c00' }}>
                        ❌ {logoError}
                      </div>
                    )}

                    <LogoOnlyForm
                      onSubmit={async (videoFile, logoFile, position, size, opacity) => {
                        try {
                          setIsLogoProcessing(true);
                          setLogoError(null);

                          // Create FormData with video, logo, and settings
                          const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8081';
                          const formData = new FormData();
                          formData.append('video', videoFile);
                          formData.append('logo', logoFile);
                          formData.append('position', position);
                          formData.append('size', size);
                          formData.append('opacity', opacity.toString());

                          // Call the add-logo-to-video endpoint
                          const response = await fetch(`${API_BASE_URL}/add-logo-to-video`, {
                            method: 'POST',
                            body: formData,
                            credentials: 'include'
                          });

                          if (!response.ok) {
                            const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
                            throw new Error(errorData.error || `HTTP ${response.status}`);
                          }

                          // Download the result
                          const blob = await response.blob();
                          const url = window.URL.createObjectURL(blob);
                          const a = document.createElement('a');
                          a.href = url;
                          a.download = `${videoFile.name.split('.')[0]}_with_logo.mp4`;
                          document.body.appendChild(a);
                          a.click();
                          window.URL.revokeObjectURL(url);
                          document.body.removeChild(a);

                          // Show success message
                          alert('✅ ' + t('logoOnly.successMessage'));

                        } catch (err: unknown) {
                          setLogoError(err instanceof Error ? err.message : 'Unknown error adding logo');
                        } finally {
                          setIsLogoProcessing(false);
                        }
                      }}
                      isProcessing={isLogoProcessing}
                    />
                  </div>
                )}

                {selectedAdvancedFeature === 'video-cutter' && (
                  <div className="advanced-feature-content">
                    <h3>✂️ {t('advanced.videoCutter')}</h3>
                    <p className="feature-description">{t('advanced.videoCutterDescription')}</p>

                    <VideoCutterForm
                      onSubmit={async (file, startTime, endTime) => {
                        try {
                          const formData = new FormData();
                          formData.append('video', file);
                          formData.append('start_time', startTime);
                          formData.append('end_time', endTime);

                          const apiBaseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8081';
                          const response = await fetch(`${apiBaseUrl}/cut-video`, {
                            method: 'POST',
                            body: formData,
                            credentials: 'include'
                          });

                          if (!response.ok) {
                            throw new Error('Video cutting failed');
                          }

                          // Download the cut video
                          const blob = await response.blob();
                          const url = window.URL.createObjectURL(blob);
                          const a = document.createElement('a');
                          a.href = url;
                          a.download = `cut_video_${startTime.replace(/:/g, '')}_${endTime.replace(/:/g, '')}.mp4`;
                          document.body.appendChild(a);
                          a.click();
                          document.body.removeChild(a);
                          window.URL.revokeObjectURL(url);
                        } catch (error) {
                          alert('Error cutting video. Please try again.');
                        }
                      }}
                      isProcessing={isProcessing}
                    />
                  </div>
                )}

                {selectedAdvancedFeature === 'embed-subtitles' && (
                  <div className="advanced-feature-content">
                    <h3>📝 {t('advanced.embedSubtitles')}</h3>
                    <p className="feature-description">{t('advanced.embedSubtitlesDescription')}</p>

                    <EmbedSubtitlesForm
                      watermarkConfig={watermarkConfig}
                      onSubmit={async (videoFile, subtitlesFile, subtitlesText, includeLogo, position, size, opacity) => {
                        try {
                          const formData = new FormData();
                          formData.append('video', videoFile);

                          if (subtitlesFile) {
                            formData.append('srt_file', subtitlesFile);
                          } else {
                            formData.append('srt_text', subtitlesText);
                          }

                          formData.append('include_logo', includeLogo.toString());
                          formData.append('logo_position', position);
                          formData.append('logo_size', size);
                          formData.append('logo_opacity', opacity.toString());

                          const apiBaseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8081';
                          const response = await fetch(`${apiBaseUrl}/embed-subtitles`, {
                            method: 'POST',
                            body: formData,
                            credentials: 'include'
                          });

                          if (!response.ok) {
                            throw new Error('Embedding subtitles failed');
                          }

                          // Download the video
                          const blob = await response.blob();
                          const url = window.URL.createObjectURL(blob);
                          const a = document.createElement('a');
                          a.href = url;
                          a.download = `video_with_subtitles_${videoFile.name}`;
                          document.body.appendChild(a);
                          a.click();
                          document.body.removeChild(a);
                          window.URL.revokeObjectURL(url);
                        } catch (error) {
                          alert('Error embedding subtitles. Please try again.');
                        }
                      }}
                      isProcessing={isProcessing}
                    />
                  </div>
                )}

                {selectedAdvancedFeature === 'video-merger' && (
                  <div className="advanced-feature-content">
                    <h3>🔗 {t('advanced.videoMerger')}</h3>
                    <p className="feature-description">{t('advanced.videoMergerDescription')}</p>

                    <VideoMergerForm
                      onSubmit={async (video1, video2) => {
                        try {
                          const formData = new FormData();
                          formData.append('video1', video1);
                          formData.append('video2', video2);

                          const apiBaseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8081';
                          const response = await fetch(`${apiBaseUrl}/merge-videos`, {
                            method: 'POST',
                            body: formData,
                            credentials: 'include'
                          });

                          if (!response.ok) {
                            throw new Error('Video merging failed');
                          }

                          // Download the merged video
                          const blob = await response.blob();
                          const url = window.URL.createObjectURL(blob);
                          const a = document.createElement('a');
                          a.href = url;
                          a.download = `merged_${video1.name.split('.')[0]}_${video2.name.split('.')[0]}.mp4`;
                          document.body.appendChild(a);
                          a.click();
                          document.body.removeChild(a);
                          window.URL.revokeObjectURL(url);
                        } catch (error) {
                          alert('Error merging videos. Please try again.');
                        }
                      }}
                      isProcessing={isProcessing}
                    />
                  </div>
                )}

                {selectedAdvancedFeature === 'audio-extractor' && (
                  <div className="advanced-feature-content">
                    <h3>🎵 {t('advanced.audioExtractor')}</h3>
                    <p className="feature-description">{t('advanced.audioExtractorDescription')}</p>

                    <AudioExtractorForm
                      onSubmit={async (file: File, format: string) => {
                        try {
                          const formData = new FormData();
                          formData.append('video', file);
                          formData.append('format', format);

                          const apiBaseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8081';
                          const response = await fetch(`${apiBaseUrl}/extract-audio`, {
                            method: 'POST',
                            body: formData,
                            credentials: 'include'
                          });

                          if (!response.ok) {
                            const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
                            throw new Error(errorData.error || `HTTP ${response.status}`);
                          }

                          // Download the audio file
                          const blob = await response.blob();
                          const url = window.URL.createObjectURL(blob);
                          const a = document.createElement('a');
                          a.href = url;
                          a.download = `${file.name.split('.')[0]}_audio.${format}`;
                          document.body.appendChild(a);
                          a.click();
                          document.body.removeChild(a);
                          window.URL.revokeObjectURL(url);
                        } catch (error) {
                          alert('Error extracting audio. Please try again.');
                        }
                      }}
                      isProcessing={isProcessing}
                    />
                  </div>
                )}
              </div>
            </main>
          </>
        )}

        <AuthModal 
          isOpen={showAuthModal} 
          onClose={() => setShowAuthModal(false)} 
        />
      </div>
    </div>
  );
}

export default App;
