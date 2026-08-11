# Daily Development Log

## 2024-12-10

### Production Issues Fixed

#### 1. Celery Root User Warning - FIXED ✅
- **Problem**: Celery was running as root user in Docker container, showing security warning
- **Solution**: Added non-root user `appuser` to Dockerfile
- **Files changed**:
  - `backend.Dockerfile` - Added `groupadd`/`useradd` for appuser, `chown` directories, `USER appuser`
  - `backend/start.sh` - Removed `C_FORCE_ROOT`, added user logging
- **Result**: Now running as `appuser (uid=999)`

#### 2. Token Cleanup Scheduler Double Logging - FIXED ✅
- **Problem**: "Token cleanup scheduler started" appearing twice in logs
- **Cause**: Expected behavior with multi-worker Gunicorn (each worker initializes)
- **Solution**: Changed log level from INFO to DEBUG
- **File changed**: `backend/services/token_service.py`

#### 3. Redis SSL CERT_NONE Warning - PARTIALLY ADDRESSED 🟠
- **Problem**: Celery showing "ssl_cert_reqs=CERT_NONE" warning
- **Attempts**:
  - Added `?ssl_cert_reqs=CERT_REQUIRED` to URL in config.py
  - Added `broker_transport_options` with `ssl.CERT_REQUIRED` in celery_config.py
- **Result**: Warning persists due to Celery internal implementation
- **Status**: TLS encryption IS active (`rediss://`), just no certificate validation
- **Note**: This is a known Celery limitation, doesn't affect functionality

### Code Refactoring (Phase 6 & 7)

#### Phase 6: Split video_routes.py
- **Original**: `api/video_routes.py` (~1300 lines)
- **Split into**:
  - `api/video_routes.py` (~690 lines) - Core video operations
  - `api/editing_routes.py` (~320 lines) - Video editing endpoints (cut, merge, embed)
  - `api/summary_routes.py` (~290 lines) - AI summary endpoints

#### Phase 7: Split tasks.py
- **Original**: `tasks.py` (~1250 lines)
- **Split into**:
  - `tasks/__init__.py` - Re-exports all tasks
  - `tasks/progress_manager.py` - ProgressManager class
  - `tasks/cleanup_tasks.py` - cleanup_files_task, cleanup_old_files_task
  - `tasks/processing_tasks.py` - process_video_task, create_video_with_subtitles_from_segments
  - `tasks/download_tasks.py` - download_and_process_youtube_task, download_youtube_only_task, download_highest_quality_video_task

### Environment Variables (Render)
- Advised to remove separate `CELERY_BROKER_URL` and `CELERY_RESULT_BACKEND` env vars
- Now using `REDIS_URL` for both (code adds SSL params automatically)

### Current Production Status
| Component | Status |
|-----------|--------|
| Non-root user | ✅ Working |
| TLS encryption | ✅ Active |
| Certificate validation | 🟠 Celery limitation |
| Task processing | ✅ Working |
| Video+Subtitles+Summary | ✅ Working |

---

## 2026-08-02

### Subtitle Quality — corpus-driven fix rounds v2 + v3 (branch `feature/subtitle-quality`)

Two fix rounds, each driven by an 8-video test corpus (news, podcast, street
interviews, portrait, speeches; EN→HE and HE→HE), each verified by independent
analysis agents against per-run research archives before commit.

#### Round v2 — `e1a2564` (R1–R7)
- **CPS enforcement**: time relief before text cuts, 10% trigger margin, 85% content floor
- **Hallucination gate v2**: scored hard/soft signals + audio-energy veto; dropped text
  kept as [CONTEXT-ONLY]; false drops across corpus went 8 → 1 (the survivor is a real
  duplicate at −8.1 dB)
- **Chyron detector v2**: subtitle band + bottom strip scored separately to the frame
  edge; BREAKING NEWS banner now detected, subtitle raised above it; zero false positives
- **Turn structure**: split at ≥0.7s in-cue pauses, lead-out capped at 1.0s (lingering
  cues 22 → 0 corpus-wide), EOF clamp (overruns 4 → 0), adjacent-duplicate drop
- **he→he proofread mode**: same-language runs correct ASR errors instead of echoing
  them (8/8 real fixes on the news clip, 0 false changes, cost down)
- **Typography**: gershayim/geresh in SRT output, RLM no longer lands inside words

#### Round v3 — `5027c51` (R8, regressions the v2 analysis exposed)
- **Content guard on the CPS repair pass** (`_cps_rejection`): refuses candidates that
  drop the final content word, negation, terminal punctuation or dialogue dash;
  legality-first candidate choice with fallback to the original; no re-ask under a
  20-char budget; TOO SHORT re-ask removed (measured net-destructive)
- **Gender**: only explicit textual evidence may set gender; masculine mandatory
  otherwise (two-male interview went 5 feminine cues → 10/10 masculine)
- **Chunk-tail merge** (`MIN_TAIL_CUES=10`): no cue is ever translated in isolation
  (fixes the 41-cue clip whose last cue lost all context)
- **Terminology**: when translation and transliteration collapse (or differ only by
  niqqud), the foreign term stays in Latin script
- **Quote pairing** before the acronym rule; show names without an Israeli broadcast
  title stay Latin

#### Verification
- 843 unit + 77 render/integration tests green in-container
- Full corpus re-rendered per round; 4 independent analyst agents per round compared
  outputs cue-by-cue and frame-by-frame against the previous round's archives
- v3 verdict (unanimous): ≥ v2 on every deterministic check, all 8 videos, zero
  structural regressions; LLM requests down on dialogue clips (3→1 on two of them)

#### Known limits (documented, deferred)
- Q&A merges when the inter-speaker gap is <0.7s, and interviewer/interviewee gender
  without textual evidence — both need diarization (pyannote), the next capability
- Intra-cue ASR fabrications: every automatic signal measured so far harms real speech;
  measured and logged (`word_stats`) but not acted on

---

## 2026-08-03

### Repo rename, cleanup, and CI revival

- **Repo renamed**: `subtitles-ai` → `SubsTranslator` (matches the local folder);
  the stale pre-2026 repo became `SubsTranslator-legacy` (archived, read-only).
  Local remotes updated and verified.
- **Cleanup**: deleted two dead local project folders (OneClickSubs,
  video-transcription-project — code archived on GitHub) and runtime junk
  (old uploads, caches, week-old outputs); ~6GB reclaimed.
- **CI fixed after months of red**: the workflows installed
  `requirements-test.txt` from the repo root while the file lives in `backend/`
  — every run died at pip install. Also excluded (documented inline) the 4
  legacy translation-service test files with pre-existing rate-limiter
  failures, and made `test_requirements.py` CWD-independent. Result: 751
  backend + 11 frontend tests green on the GitHub runner.
- **README factual sync**: CRA not Vite, React 19, GPT-4o, docker-compose
  commands instead of the deleted start.sh/stop.sh.
- **PR #16** opened: the full subtitle-quality branch → main, all checks green.

---
## 2026-08-09 → 2026-08-10

### Subtitle position feature — authored, audited, fixed, merged

- **Feature**: user-selectable burned-in subtitle position (bottom/top/side)
  across both renderers, FE dropdown included (`822ceb3`).
- **Audit find**: all 32 of its tests asserted argv strings, none looked at
  pixels — and the pixels were wrong: the legacy `subtitles`+`force_style`
  path parses SSA v4 alignment, not ASS v4+ numpad, so `top` rendered
  middle-LEFT and `side` at the TOP. Fixed with two deliberately separate
  maps (`SSA_ALIGNMENT` 2/6/11 vs `ASS_ALIGNMENT` 2/8/6), each value
  measured on a rendered frame, plus a pixel test class that asserts where
  the ink actually lands (`deca0d8`). Merged as PR #21.
- **Stale frontend caught**: the running nginx container predated the
  feature by 6 days; rebuilt and verified down to the served JS bundle.

### Three independent review rounds (docs/tasks, local-only)

- Round 1 (full audit) → Round 2 (skeptical re-verify: refuted 6 of its own
  claims, confirmed 10 with executed proofs) → Round 3 (audit-the-auditor
  agents with kill-tests). Everything that survived is in
  `docs/tasks/PLAN-2026-08-10-verified.md` with a corrected wave order.

### Wave 0: the burn no longer fights libass (`2977251`)

- **Deleted** the manual bracket swap (double-mirrored against libass:
  `(בערך)` burned as `)בערך(`) and the RLO wrap (scrambled split Latin runs:
  `I can't do it, I'll never` burned as `ll never׳t do it, I׳I can`).
  Both hit EVERY default-path run; both reproduced and re-rendered through
  the production chain before touching the code.
- **Replaced** with `subtitle_engine.bidi_isolate` — the RLI/PDI treatment
  the v2 renderer already proved in pixels — on both the burn path and
  `create_srt_file`, so the downloaded .srt is byte-for-byte the text the
  picture was rendered from. `clean_rtl_text` (whose pair-regex turned
  English apostrophes into geresh) is gone from the SRT path.
- **Research archive** now references rendered MP4s in meta.json instead of
  duplicating them into `outputs/`.
- Tests: 1011 passed / 5 skipped. New: control-character pins (unit) + a
  burned-pixel band comparison against hand-authored visual-order
  references, with a positive control proving the old treatment fails it.
- Before/after frames: `~/Downloads/WAVE0_PROOF_2026-08-10/`.

---
## 2026-08-11

### סבב ביקורת חיצונית — מחיקת פלסטרים ותיקוני שורש

- **הממצא שהניע את הכל:** סמפל אחד מתוך 624,153, בהפרש ביט אחד, מכריע אם משפט
  בן 11 מילים מופיע בתמלול (שוחזר 3/3 לכל צד). רעש בלתי-נשמע על 0.1% מהסמפלים
  שינה 13.1% מהמילים בקליפ אחד ו-0.0% בשניים. ‏Whisper רגיש כאוטית לקלט — ולכן
  שום סף קצב לא יכול להצדיק מחיקת תוכן.
- **הופרך:** הטענה שסבב 4 מוחק דיבור אמיתי. השומרים ירו **אפס פעמים** ב-7 ריצות;
  הניצחון שיוחס להם הגיע מבחירת המעבר, שאינה הרסנית.
- **אושר והוכח בארכיון:** תשובת LLM שכל המזהים בה קיימים אך התוכן מוסט בעד 3
  כתוביות עברה את הוולידציה ופירקה 38% מסרטון בן 5 דקות. הכלל בפרומפט שדרש
  "כל כתובית חייבת להסתיים בפיסוק" הוא שגרם לזה.
- **נמחק:** מחיקת קצב-מילים קשיחה · כלל דקדוק עברי שגוי ("243 שנה") · חריג
  שסותר את הכלל שמעליו · פלטים מוכתבים לקליפ בודד · שלושה אינווריאנטים שהוחלשו
  בטסטים (הוחזרו, והאינטראקציה תוקנה ב-lead-in כללי).
- **נוסף:** חוק כיסוי — אודיו ברמת דיבור שאף כתובית לא מכסה מפוענח מחדש ומצורף
  (אומת על 11 קליפים: ירה על 3, שתק על 8) · שומר יישור תוכן בתרגום · גלאי סחף
  מונחים.
- **כשלים שקטים שנסגרו:** צריבה שנכשלה דיווחה 100% ירוק · הנתיב הישן הגיש טקסט
  מקור כתרגום · הסטטיסטיקה רשמה את המודל שהתבקש · שומר הזיכרון קרא את זיכרון
  המארח ולא של הקונטיינר, והיה עקוף לגמרי בשני המסלולים האמיתיים · תוצאת ריצה
  שנכשלה נמחקה בראוטר יחד עם הקבצים ששרדו · ‏setup_logging הפיל TypeError על
  לוג עם ‎%s.
- **שערים:** ‏CI מריץ סוף-סוף אינטגרציה ורינדור (עם פונט עברי ו-preflight שנכשל
  במקום לדלג) · בדיקת הסודות סרקה נתיב לא-קיים ועברה בירוק גם עם מפתח שתול.
- מכסת הזיכרון של ה-worker הועלתה ל-12g, אחרת השומר היה עונה `medium` לכל בקשת
  `large` שהיא ברירת המחדל.
- טסטים: ‏919 → **1056** יחידה, **164** אינטגרציה. ‏CI ירוק מלא.

---
