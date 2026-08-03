# 🚀 המדריך המלא לעבודה עם faster-whisper

## 📋 **מה זה faster-whisper?**

faster-whisper הוא יישום מחדש של מודל Whisper של OpenAI באמצעות CTranslate2, שהוא מנוע inference מהיר למודלים של Transformer. היישום הזה מהיר עד פי 4 מ-openai/whisper עם אותה דיוק ומשתמש בפחות זיכרון.

---

## 🛠️ **התקנה מהירה**

### **⚠️ הערות חשובות לגבי Python:**
- **Python 3.13:** לא נתמך כרגע (ctranslate2 לא תומך)
- **מומלץ:** Python 3.12 (הגרסה שהפרויקט נבנה ונבדק עליה)

### **שלב 1: התקנה בסיסית**
```bash
# התקנה סטנדרטית
pip install faster-whisper

# או התקנה מהקוד החדש ביותר
pip install --force-reinstall "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/refs/heads/master.tar.gz"
```

### **שלב 2: תלויות נוספות (אופציונלי)**
```bash
# עבור GPU (NVIDIA) - בדוק תאימות CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# עבור קבצי שמע נוספים
pip install librosa soundfile
```

---

## ⚡ **שימוש בסיסי מוטב**

### **דוגמה פשוטה - מיטבית לCPU:**
```python
from faster_whisper import WhisperModel

# הגדרות מיטביות לCPU
model_size = "base"  # או "small", "medium" לפי הצורך
model = WhisperModel(
    model_size, 
    device="cpu", 
    compute_type="int8",    # 🔥 פי 2 מהיר יותר!
    num_workers=1,          # מיטבי למשימות כבדות
    download_root="./models"  # cache מקומי
)

# תמלול מיטבי
segments, info = model.transcribe(
    "audio.mp3", 
    beam_size=5,                    # ברירת מחדל ב-faster-whisper (לא 1!)
    language="auto",                # זיהוי אוטומטי
    condition_on_previous_text=False,  # מהיר יותר
    vad_filter=True,                # סינון רעש טוב יותר
    vad_parameters=dict(min_silence_duration_ms=500)
)

print(f"שפה מזוהה: {info.language} ({info.language_probability:.2f})")

# הדפסת תוצאות
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

### **דוגמה מתקדמת עם אופטימיזציות:**
```python
from faster_whisper import WhisperModel
import time

class OptimizedWhisper:
    def __init__(self, model_size="base", device="cpu"):
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type="int8" if device == "cpu" else "float16",
            num_workers=1,
            local_files_only=False,
            download_root="./whisper_models"
        )
    
    def transcribe_optimized(self, audio_path, target_language="he"):
        start_time = time.time()
        
        # הגדרות מיטביות
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5,                      # ברירת מחדל ב-faster-whisper
            best_of=1,                        # ללא חיפושים נוספים
            temperature=0.0,                  # deterministic
            condition_on_previous_text=False, # מהיר יותר
            compression_ratio_threshold=2.4,  # זיהוי hallucinations
            log_prob_threshold=-1.0,          # סינון תוצאות חלשות
            no_speech_threshold=0.6,          # זיהוי שקט
            vad_filter=True,                  # חשוב לאיכות
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=400
            )
        )
        
        processing_time = time.time() - start_time
        
        # המרה לרשימה (segments הוא generator)
        segments_list = list(segments)
        
        print(f"⏱️ זמן עיבוד: {processing_time:.2f} שניות")
        print(f"🎯 שפה: {info.language} (ביטחון: {info.language_probability:.2f})")
        print(f"📊 {len(segments_list)} segments")
        
        return segments_list, info
    
    def save_to_file(self, segments, output_path):
        """שמירה לקובץ טקסט"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for segment in segments:
                f.write(f"{segment.text.strip()}\n")
        print(f"✅ נשמר ב: {output_path}")

# שימוש
whisper = OptimizedWhisper(model_size="base")
segments, info = whisper.transcribe_optimized("audio.mp3")
whisper.save_to_file(segments, "transcript.txt")
```

---

## 🎯 **הגדרות מיטביות לפי מקרה שימוש**

### **🏃‍♂️ מהירות מקסימלית (real-time):**
```python
model = WhisperModel(
    "tiny",                    # מודל הכי קטן
    device="cpu",
    compute_type="int8",       # דחיסה מקסימלית
    num_workers=1
)

segments, info = model.transcribe(
    audio_path,
    beam_size=1,              # הכי מהיר (לא ברירת המחדל!)
    best_of=1,
    temperature=0.0,
    language="he",            # הגדרה ידנית (מהיר יותר)
    condition_on_previous_text=False,
    vad_filter=False          # מהיר יותר אבל פחות איכות
)
```

### **🎯 איכות מקסימלית:**
```python
model = WhisperModel(
    "large-v3",               # מודל הכי טוב
    device="cpu",
    compute_type="int8",      # עדיין מהיר
    num_workers=1
)

segments, info = model.transcribe(
    audio_path,
    beam_size=5,              # ברירת מחדל ב-faster-whisper
    best_of=5,                # מספר ניסיונות
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8],  # מספר טמפרטורות
    condition_on_previous_text=True,         # הקשר
    vad_filter=True,          # סינון רעש
    vad_parameters=dict(
        min_silence_duration_ms=300,
        speech_pad_ms=400
    )
)
```

### **⚖️ איזון מושלם (מומלץ):**
```python
model = WhisperModel(
    "base",                   # איזון טוב
    device="cpu",
    compute_type="int8",
    num_workers=1
)

segments, info = model.transcribe(
    audio_path,
    beam_size=2,              # איזון מהירות/איכות
    temperature=0.0,
    language="auto",          # זיהוי אוטומטי
    condition_on_previous_text=False,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500)
)
```

---

## 🚀 **אופטימיזציות מתקדמות**

### **1. 📦 Batching עבור קבצים מרובים (גרסאות חדשות בלבד):**
```python
# ⚠️ זמין רק בגרסאות מאוד חדשות של faster-whisper
try:
    from faster_whisper import BatchedInferencePipeline
    
    # יצירת pipeline מקבילי
    batched_model = BatchedInferencePipeline(
        model=model,
        chunk_length=30,          # אורך חלקים
        stride_length=5,          # חפיפה
        batch_size=16             # כמות מקבילית
    )
    
    # עיבוד מקבילי
    segments, info = batched_model.transcribe("long_audio.mp3")
except ImportError:
    print("BatchedInferencePipeline לא זמין בגרסה זו")
```

### **2. 🎛️ ניהול זיכרון חכם:**
```python
import gc
import torch

class MemoryEfficientWhisper:
    def __init__(self, model_size="base"):
        self.model_size = model_size
        self.model = None
    
    def load_model(self):
        if self.model is None:
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                num_workers=1
            )
    
    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()  # ניקוי זיכרון
    
    def transcribe_with_cleanup(self, audio_path):
        try:
            self.load_model()
            segments, info = self.model.transcribe(audio_path)
            return list(segments), info
        finally:
            self.unload_model()  # ניקוי אוטומטי
```

### **3. 📈 ניטור ביצועים:**
```python
import psutil
import time

def monitor_performance(func):
    def wrapper(*args, **kwargs):
        # מדידת זיכרון לפני
        memory_before = psutil.virtual_memory().used / 1024 / 1024  # MB
        cpu_before = psutil.cpu_percent()
        start_time = time.time()
        
        # הרצה
        result = func(*args, **kwargs)
        
        # מדידת ביצועים אחרי
        end_time = time.time()
        memory_after = psutil.virtual_memory().used / 1024 / 1024
        cpu_after = psutil.cpu_percent()
        
        print(f"⏱️ זמן: {end_time - start_time:.2f}s")
        print(f"💾 זיכרון: {memory_after - memory_before:.1f}MB")
        print(f"🔥 CPU: {(cpu_before + cpu_after) / 2:.1f}%")
        
        return result
    return wrapper

@monitor_performance
def transcribe_with_monitoring(audio_path):
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path)
    return list(segments), info
```

---

## 🔧 **טיפים מעשיים לפרודקשן**

### **1. ⚡ cache מודלים:**
```python
# Singleton pattern למודל
class WhisperSingleton:
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, model_size="base"):
        if self._model is None:
            self._model = WhisperModel(
                model_size,
                device="cpu", 
                compute_type="int8"
            )
        return self._model

# שימוש
whisper = WhisperSingleton()
model = whisper.get_model()
```

### **2. 🛡️ טיפול בשגיאות:**
```python
def safe_transcribe(audio_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            model = WhisperModel("base", device="cpu", compute_type="int8")
            segments, info = model.transcribe(audio_path)
            return list(segments), info
        except Exception as e:
            print(f"ניסיון {attempt + 1} נכשל: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(1)  # המתנה לפני ניסיון חוזר
```

### **3. 🎚️ הגדרות מותאמות לשפה:**
```python
LANGUAGE_CONFIGS = {
    "he": {  # עברית
        "model_size": "large-v3",    # דיוק טוב יותר לעברית
        "beam_size": 3,
        "condition_on_previous_text": True,  # חשוב לעברית
        "vad_filter": True
    },
    "en": {  # אנגלית
        "model_size": "base",
        "beam_size": 2,              # איזון טוב
        "condition_on_previous_text": False,
        "vad_filter": True
    },
    "ar": {  # ערבית
        "model_size": "large-v3",
        "beam_size": 5,
        "condition_on_previous_text": True,
        "vad_filter": True
    }
}

def transcribe_by_language(audio_path, language="auto"):
    config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["en"])
    
    model = WhisperModel(
        config["model_size"],
        device="cpu",
        compute_type="int8"
    )
    
    segments, info = model.transcribe(
        audio_path,
        beam_size=config["beam_size"],
        language=language if language != "auto" else None,
        condition_on_previous_text=config["condition_on_previous_text"],
        vad_filter=config["vad_filter"]
    )
    
    return list(segments), info
```

---

## 📊 **השוואת ביצועים מצופים**

### **מהירות לפי מודל (וידאו 10 דקות):**

| **מודל** | **זמן עיבוד** | **איכות** | **זיכרון** | **מומלץ ל** |
|-----------|----------------|-----------|-------------|-------------|
| `tiny` | 2-3 דקות | בסיסי | 200MB | real-time |
| `base` | 4-5 דקות | טוב | 400MB | **עבודה יומיומית** |
| `small` | 7-8 דקות | טוב מאוד | 600MB | איכות גבוהה |
| `medium` | 12-15 דקות | מעולה | 1GB | עיבוד מקצועי |
| `large-v3` | 20-25 דקות | הטוב ביותר | 2GB | דיוק מקסימלי |

### **אופטימיזציות מצטברות:**
- **`compute_type="int8"`**: +50% מהירות ✅
- **`beam_size=1`**: +60% מהירות (במקום ברירת מחדל 5) ✅
- **`vad_filter=True`**: +20% איכות ✅
- **`condition_on_previous_text=False`**: +15% מהירות ✅

---

## 🎯 **ממצאים חשובים - עדכון מדויק:**

### **⚠️ הבדלים חשובים מ-OpenAI Whisper:**
1. **beam_size:** ברירת מחדל ב-faster-whisper הוא **5** (לא 1 כמו ב-OpenAI)
2. **השוואה הוגנת:** להשוואת מהירות, השתמש ב-`beam_size=1`
3. **BatchedInferencePipeline:** זמין רק בגרסאות מאוד חדשות
4. **Python 3.13:** לא נתמך כרגע

### **📊 השוואת beam_size:**
```python
# מהיר ביותר (השוואה הוגנת ל-OpenAI Whisper)
segments, info = model.transcribe("audio.mp3", beam_size=1)

# איכות טובה יותר (ברירת מחדל ב-faster-whisper)
segments, info = model.transcribe("audio.mp3", beam_size=5)

# איזון מושלם
segments, info = model.transcribe("audio.mp3", beam_size=2)
```

---

## 🎯 **המלצות לפרויקט שלכם**

### **להחלפה במערכת הנוכחת:**
```python
# במקום:
import whisper
model = whisper.load_model("base")
result = model.transcribe(audio_path)

# השתמש ב:
from faster_whisper import WhisperModel
model = WhisperModel("base", device="cpu", compute_type="int8")
segments, info = model.transcribe(
    audio_path, 
    beam_size=1,                      # להשוואה הוגנת עם OpenAI
    condition_on_previous_text=False,
    vad_filter=True
)

# המרה לפורמט דומה
result = {
    'text': ' '.join([seg.text for seg in segments]),
    'segments': [
        {
            'start': seg.start,
            'end': seg.end, 
            'text': seg.text
        } for seg in segments
    ],
    'language': info.language
}
```

**התוצאה:** **פי 2-4 מהיר יותר** עם **אותה איכות בדיוק**! 🚀

---

## ✅ **סיכום הערכה:**

### **מה שמדויק במדריך:**
1. ✅ **compute_type="int8"** מהיר פי 2 ב-CPU
2. ✅ **vad_filter=True** שיפור איכות משמעותי
3. ✅ **מהירות פי 4** מ-OpenAI Whisper (עם הגדרות נכונות)
4. ✅ **תמיכה ב-quantization**

### **תיקונים חשובים:**
1. ⚠️ **beam_size:** ברירת מחדל היא 5 (לא 1)
2. ⚠️ **Python 3.13:** לא נתמך כרגע
3. ⚠️ **BatchedInferencePipeline:** זמין רק בגרסאות חדשות
4. ⚠️ **beam_size=1:** להשוואה הוגנת עם OpenAI Whisper

### **המלצה סופית:**
להשגת המהירות המקסימלית תוך שמירה על איכות טובה:
- **מודל:** `base` או `small`
- **beam_size:** `1` למהירות, `2` לאיזון, `5` לאיכות
- **compute_type:** `int8`
- **vad_filter:** `True`
