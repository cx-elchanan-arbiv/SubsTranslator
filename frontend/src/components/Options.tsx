import { useEffect, useState } from 'react';
import { useTranslation } from '../i18n/TranslationContext';
import { useI18n } from '../i18n/I18nProvider';
import type {
  WhisperModel,
  TranslationService as TranslationServiceType,
  SubtitleQualityFlags,
  TranslationStyle,
} from '../types';

interface TranslationService {
  name: string;
  available: boolean;
  description: string;
}

interface WhisperModelOption {
  name: string;
  description: string;
  time_estimate: string;
  accuracy: string;
  best_for: string;
  compute_type: string;
  // Restriction info for hosted mode
  restricted?: boolean;
  restrictedReason?: string;
  tier?: 'free' | 'pro';
}

interface OptionsProps {
  autoCreateVideo: boolean;
  onAutoCreateVideoChange: (checked: boolean) => void;
  whisperModel: WhisperModel;
  onWhisperModelChange: (model: WhisperModel) => void;
  translationService: TranslationServiceType;
  onTranslationServiceChange: (service: TranslationServiceType) => void;
  transcriptionOnly: boolean;
  onTranscriptionOnlyChange: (checked: boolean) => void;
  subtitleFlags: SubtitleQualityFlags;
  onSubtitleFlagsChange: (flags: SubtitleQualityFlags) => void;
  disabled: boolean;
  activeTab?: 'upload' | 'youtube';
}

const Options: React.FC<OptionsProps> = ({
  autoCreateVideo,
  onAutoCreateVideoChange,
  whisperModel,
  onWhisperModelChange,
  translationService,
  onTranslationServiceChange,
  transcriptionOnly,
  onTranscriptionOnlyChange,
  subtitleFlags,
  onSubtitleFlagsChange,
  disabled,
  activeTab = 'youtube',
}) => {
  const { t } = useTranslation();
  const { language } = useI18n();
  const [translationServices, setTranslationServices] = useState<Record<string, TranslationService>>({});
  const [whisperModels, setWhisperModels] = useState<Record<string, WhisperModelOption>>({});

  useEffect(() => {
    const apiBaseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8081';

    // Load translation services with current language
    fetch(`${apiBaseUrl}/translation-services`, {
      headers: {
        'Accept-Language': String(language)
      }
    })
      .then(res => res.json())
      .then(data => {
        setTranslationServices(data);
      })
      .catch(err => {
        console.error('Failed to load translation services:', err);
        // Fallback to default services
        setTranslationServices({
          google: { name: 'Google Translate', available: true, description: t('options.googleDescription') || 'Free translation service' },
          openai: { name: 'OpenAI (GPT-4o)', available: false, description: t('options.openaiDescription') || 'OpenAI API key required' }
        });
      });

    // Load Whisper models
    fetch(`${apiBaseUrl}/whisper-models`)
      .then(res => res.json())
      .then(data => {
        // Apply translations to server data
        const serverModels = data.models || {};
        const translatedModels: any = {};

        Object.keys(serverModels).forEach(key => {
          const model = serverModels[key];
          translatedModels[key] = {
            ...model,
            name: `${model.name.split('(')[0].trim()} (${t(`whisperModels.${key}`)})`,
            description: t(`modelDetails.${key === 'base' ? 'goodBalance' :
                           key === 'medium' ? 'perfectBalance' : 'maximumAccuracy'}`),
            time_estimate: t(`modelDetails.${key}Time`),
            accuracy: t(`modelDetails.${key}Accuracy`),
            best_for: t(`modelDetails.${key}BestFor`),
            compute_type: t('modelDetails.computeTypeInt8'),
            // Preserve restriction info from server
            restricted: model.restricted || false,
            restrictedReason: model.restrictedReason || null,
            tier: model.tier || 'free',
          };
        });

        setWhisperModels(translatedModels);
      })
      .catch(err => {
        console.error('Failed to load whisper models:', err);
        // Fallback to default models (all using int8 on CPU)
        setWhisperModels({
          base: {
            name: `Base (${t('whisperModels.base')})`,
            description: t('modelDetails.goodBalance'),
            time_estimate: t('modelDetails.baseTime'),
            accuracy: t('modelDetails.baseAccuracy'),
            best_for: t('modelDetails.baseBestFor'),
            compute_type: t('modelDetails.computeTypeInt8')
          },
          medium: {
            name: `Medium (${t('whisperModels.medium')})`,
            description: t('modelDetails.perfectBalance'),
            time_estimate: t('modelDetails.mediumTime'),
            accuracy: t('modelDetails.mediumAccuracy'),
            best_for: t('modelDetails.mediumBestFor'),
            compute_type: t('modelDetails.computeTypeInt8')
          },
          large: {
            name: `Large (${t('whisperModels.large')})`,
            description: t('modelDetails.maximumAccuracy'),
            time_estimate: t('modelDetails.largeTime'),
            accuracy: t('modelDetails.largeAccuracy'),
            best_for: t('modelDetails.largeBestFor'),
            compute_type: t('modelDetails.computeTypeInt8')
          },
          gemini: {
            name: 'Gemini 2.5 Flash (⚡ ניסיוני)',
            description: 'AI מתקדם מאת Google - מהיר במיוחד',
            time_estimate: 'מהיר מאוד (1-2 דקות)',
            accuracy: 'ניסיוני (יכול להיות פחות מדויק)',
            best_for: 'סרטונים קצרים מ-YouTube (עד 15 דקות)',
            compute_type: 'Cloud API'
          }
        });
      });
  }, [t, language]);

  // Filter models based on active tab - Gemini only works with YouTube
  const filteredModels = Object.entries(whisperModels).filter(([key]) => {
    // Hide Gemini on upload tab since it only works with YouTube URLs
    if (key === 'gemini' && activeTab === 'upload') {
      return false;
    }
    return true;
  });

  return (
    <div className="options">
      {whisperModel === 'gemini' && activeTab === 'youtube' && (
        <div className="model-warning" style={{
          padding: '10px',
          marginBottom: '15px',
          backgroundColor: '#fff3cd',
          border: '1px solid #ffc107',
          borderRadius: '4px',
          color: '#856404',
          fontSize: '0.9em'
        }}>
          ⚠️ <strong>Gemini בשלב ניסיוני:</strong><br/>
          • מומלץ לסרטונים קצרים (עד 15 דקות)<br/>
          • עובד רק עם YouTube URLs<br/>
          • אם נכשל, המערכת תעבור אוטומטית ל-Whisper
        </div>
      )}
      <div className="checkbox-option">
        <input
          type="checkbox"
          id="autoCreateVideo"
          checked={autoCreateVideo}
          onChange={(e) => onAutoCreateVideoChange(e.target.checked)}
          disabled={disabled}
        />
        <label htmlFor="autoCreateVideo">{t.autoCreateVideo}</label>
      </div>
      <div className="model-selection">
        <label>{t.whisperModel}</label>
        <div className="model-controls">
          <select
            value={whisperModel}
            onChange={(e) => {
              const selectedModel = whisperModels[e.target.value];
              // Don't allow selecting restricted models
              if (selectedModel?.restricted) {
                return;
              }
              onWhisperModelChange(e.target.value as WhisperModel);
            }}
            disabled={disabled}
          >
            {filteredModels.map(([key, model]) => (
              <option
                key={key}
                value={key}
                disabled={model.restricted}
                style={{
                  color: model.restricted ? '#999' : 'inherit',
                  fontStyle: model.restricted ? 'italic' : 'normal',
                }}
                title={model.restricted ? model.restrictedReason : undefined}
              >
                {model.name}
                {model.tier === 'pro' ? ` [${t('whisperModels.pro')}]` : ''}
                {model.restricted ? ` 🔒` : ''}
              </option>
            ))}
            {/* Fallback options if API fails */}
            {Object.keys(whisperModels).length === 0 && (
              <>
                {/* eslint-disable-next-line i18next/no-literal-string */}
                <option value="base">Base ({t('whisperModels.base')})</option>
                {/* eslint-disable-next-line i18next/no-literal-string */}
                <option value="medium">Medium ({t('whisperModels.medium')})</option>
                {/* eslint-disable-next-line i18next/no-literal-string */}
                <option value="large">
                  Large ({t('whisperModels.large')}) [{t('whisperModels.pro')}]
                </option>
              </>
            )}
          </select>
          {/* Show tooltip message for restricted models */}
          {whisperModels[whisperModel]?.restricted && (
            <div className="model-restriction-notice" style={{
              fontSize: '0.8em',
              color: '#d32f2f',
              marginTop: '4px',
              padding: '6px 10px',
              backgroundColor: '#ffebee',
              borderRadius: '4px',
              border: '1px solid #ffcdd2'
            }}>
              🔒 {whisperModels[whisperModel].restrictedReason}
            </div>
          )}
        </div>
        {Object.keys(whisperModels).length === 0 && (
          <div className="model-hint">
            {t.modelHint}
          </div>
        )}
      </div>
      <div className="checkbox-option transcription-only-option">
        <input
          type="checkbox"
          id="transcriptionOnly"
          checked={transcriptionOnly}
          onChange={(e) => onTranscriptionOnlyChange(e.target.checked)}
          disabled={disabled}
        />
        <label htmlFor="transcriptionOnly">{t('options.transcriptionOnly')}</label>
        <span className="option-hint">{t('options.transcriptionOnlyHint')}</span>
      </div>
      <div className={`translation-service-selection ${transcriptionOnly ? 'disabled-section' : ''}`}>
        <label>{t('options.translationService')}</label>
        <div className="model-controls">
          <select
            value={translationService}
            onChange={(e) => onTranslationServiceChange(e.target.value as TranslationServiceType)}
            disabled={disabled || transcriptionOnly}
          >
            {Object.entries(translationServices).map(([key, service]) => (
              <option
                key={key}
                value={key}
                disabled={!service.available}
                style={{
                  color: service.available ? 'inherit' : '#999',
                  fontStyle: service.available ? 'normal' : 'italic'
                }}
              >
                {service.name} {!service.available ? `(${t('status.unavailable')})` : ''}
              </option>
            ))}
          </select>
        </div>
        {translationServices[translationService] && (
          <div className="service-description" style={{
            fontSize: '0.8em',
            color: translationServices[translationService].available ? '#666' : '#d32f2f',
            marginTop: '4px'
          }}>
            {translationServices[translationService].description}
          </div>
        )}
      </div>

      {/*
        Experimental subtitle-quality toggles. Four INDEPENDENT switches so each
        improvement can be judged on its own; all default to off, which runs the
        pipeline exactly as it ran before they existed.
      */}
      <div
        className="subtitle-quality-options"
        style={{
          marginTop: '15px',
          padding: '10px 12px',
          border: '1px dashed #b0bec5',
          borderRadius: '6px',
        }}
      >
        <label style={{ fontWeight: 600 }}>{t('options.subtitleQuality')}</label>
        <div className="option-hint" style={{ fontSize: '0.8em', color: '#666', marginBottom: '8px' }}>
          {t('options.subtitleQualityHint')}
        </div>

        <div className="checkbox-option">
          <input
            type="checkbox"
            id="spottingV2"
            checked={subtitleFlags.spotting_v2}
            onChange={(e) => onSubtitleFlagsChange({ ...subtitleFlags, spotting_v2: e.target.checked })}
            disabled={disabled}
          />
          <label htmlFor="spottingV2">{t('options.spottingV2')}</label>
          <span className="option-hint" style={{ fontSize: '0.8em', color: '#666' }}>
            {t('options.spottingV2Hint')}
          </span>
        </div>

        <div className="checkbox-option">
          <input
            type="checkbox"
            id="translationV2"
            checked={subtitleFlags.translation_v2}
            onChange={(e) => onSubtitleFlagsChange({ ...subtitleFlags, translation_v2: e.target.checked })}
            disabled={disabled || transcriptionOnly}
          />
          <label htmlFor="translationV2">{t('options.translationV2')}</label>
          <span className="option-hint" style={{ fontSize: '0.8em', color: '#666' }}>
            {t('options.translationV2Hint')}
          </span>
        </div>

        {/* Style is a property of translation_v2, so it is only live while that is on. */}
        <div className={`translation-style-selection ${!subtitleFlags.translation_v2 ? 'disabled-section' : ''}`}>
          <label htmlFor="translationStyle">{t('options.translationStyle')}</label>
          <div className="model-controls">
            <select
              id="translationStyle"
              value={subtitleFlags.translation_style}
              onChange={(e) =>
                onSubtitleFlagsChange({
                  ...subtitleFlags,
                  translation_style: e.target.value as TranslationStyle,
                })
              }
              disabled={disabled || !subtitleFlags.translation_v2}
            >
              <option value="clean">{t('options.translationStyleClean')}</option>
              <option value="faithful">{t('options.translationStyleFaithful')}</option>
            </select>
          </div>
        </div>

        <div className="checkbox-option">
          <input
            type="checkbox"
            id="renderV2"
            checked={subtitleFlags.render_v2}
            onChange={(e) => onSubtitleFlagsChange({ ...subtitleFlags, render_v2: e.target.checked })}
            disabled={disabled || !autoCreateVideo}
          />
          <label htmlFor="renderV2">{t('options.renderV2')}</label>
          <span className="option-hint" style={{ fontSize: '0.8em', color: '#666' }}>
            {t('options.renderV2Hint')}
          </span>
        </div>
      </div>

    </div>
  );
};

export default Options;