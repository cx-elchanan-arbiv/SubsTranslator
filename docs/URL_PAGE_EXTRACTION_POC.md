# חילוץ וידאו מ-URL של דף (לא לינק ישיר) — מחקר + POC

> **תאריך:** 2026-06-24 · **סטטוס:** מחקר/POC, לא מומש בקוד הראשי
> **שאלה:** האם אפשר להרחיב את SubsTranslator כך שמשתמש ידביק **כתובת של דף-אתר שיש בו וידאו** (לא רק לינק ישיר ל-YouTube/Vimeo), והמערכת תמצא לבד את הוידאו, תחלץ אותו ותמשיך לתמלול/תרגום?
> **סקריפט ה-POC:** [`scripts/poc_url_extract.py`](../scripts/poc_url_extract.py)

---

## TL;DR — המסקנה

| סוג אתר | מה צריך | דוגמה (נבדק) |
|---|---|---|
| extractor ייעודי / embed מזוהה ב-HTML | **yt-dlp לבד** | YouTube, TED, Wikipedia |
| login + מדיה גלויה ב-HTML/API | **yt-dlp + cookies** | Vimeo (דורש cookies) |
| **קליפת-JS + token חתום** | **Playwright חובה** (cookies לא מספיק) | **Maven** |

**הארכיטקטורה הנכונה: hybrid, yt-dlp-first.**
- **מנוע ראשי = yt-dlp** עם ה-generic extractor — מכסה את רוב האינטרנט הציבורי בזול וסקיילבילי.
- **Playwright = fallback צר** ל-15–20% הקשים (דפי-JS מאחורי login). הוכח אמפירית שאי-אפשר לדלג עליו אם רוצים לתמוך באתרים כאלה.

---

## רקע — איך זה עובד היום מול מה שמבקשים

היום SubsTranslator מקבל **לינק ישיר לוידאו** (YouTube/Vimeo) → yt-dlp מוריד → תמלול → תרגום.

הבקשה: לקבל **URL של דף רגיל שמכיל וידאו**, למצוא את הוידאו לבד, ולהמשיך כרגיל. כולל טיפול במקרה של **כמה וידאו בדף**.

---

## מה גילה המחקר (תיעוד + קהילה)

1. **yt-dlp בנוי בדיוק לזה.** ל-yt-dlp יש "Generic Extractor" שמותאם לכל URL ומופעל אחרון אם אין extractor ייעודי. התיעוד: *"extract videos from lots of websites that embed a video from another service, may also be used to extract video from a service that it's hosting itself."* מזהה נגנים מוטמעים (YouTube, Brightcove, JW Player, Mux) ומאציל אליהם.
2. **בהשוואת production — yt-dlp עדיף לוידאו, Playwright לא כברירת-מחדל.** yt-dlp = מהיר, בלי דפדפן, 1,700+ אתרים. Playwright = גמיש ל-JS דינמי, אבל *"once you scale to hundreds/thousands of pages, things break down fast — memory, CAPTCHAs, IP bans."*
3. **האתגר האמיתי בשרת הוא auth ו-IP, לא הכלי.** `--cookies-from-browser` עובד מקומית, אבל בשרת: datacenter-IP נחסם גם עם cookies תקפים; Cloudflare מזהה ש-yt-dlp לא דפדפן; Chrome נועל את מסד-הקוקיז כשהוא פתוח (Firefox קל יותר); OAuth של אפליקציה לא עוזר (צריך browser-cookies).

מקורות:
- [yt-dlp Generic Extractor — Wiki](https://github.com/yt-dlp/yt-dlp/wiki/extractors)
- [yt-dlp Supported Sites — DeepWiki](https://deepwiki.com/yt-dlp/yt-dlp/1.2-supported-sites-and-extractors)
- [Headless scraping at scale — limitations (Browserless)](https://www.browserless.io/blog/scraping-with-playwright-a-developer-s-guide-to-scalable-undetectable-data-extraction)
- [yt-dlp 2026 guide](https://roundproxies.com/blog/yt-dlp/)
- [yt-dlp cookies / bot-detection on servers — Issue #12045](https://github.com/yt-dlp/yt-dlp/issues/12045)

---

## תוצאות ה-POC (yt-dlp, probe בלבד, `download=False`)

הורץ עם `extractor_args={"youtube":{"player_client":["android_vr"]}}` כמו בפרודקשן.

| דף | extractor שנתפס | תוצאה |
|---|---|---|
| `youtube.com/watch?v=…` | Youtube (ייעודי) | ✅ 2160p, 33 פורמטים |
| `ted.com/talks/…` (דף, לא לינק ישיר) | TedTalk (ייעודי) | ✅ + **כתוביות מובנות** ב-6+ שפות |
| `en.wikipedia.org/wiki/Wikipedia` | **Generic** (סקרייפ דף) | ✅ מצא **3 וידאו** → תרחיש "כמה סרטונים" |
| `vimeo.com/…` | Vimeo | ✗ דורש login/cookies |
| `imdb.com/title/…`, `web.dev/` | — | ✗ `Unsupported URL` (generic לא מצא) |

**תובנה על "כמה וידאו בדף":** כש-generic מוצא כמה וידאו, התוצאה חוזרת כ-`_type: "playlist"` עם `entries`. כלומר ה-API כבר מחזיר רשימה — צריך להציג למשתמש לבחור, או דיפולט חכם (הארוך/הראשי).

---

## המקרה הקשה — Maven (מחקר עומק)

נבדק: `https://maven.com/p/157a67/building-real-design-systems-with-agents`

### שלב 1 — yt-dlp אנונימי
```
ERROR: Unsupported URL
```

### שלב 2 — בדיקת ה-HTML הגולמי (curl)
```
HTTP 200 · 237885 bytes
matches (m3u8/mux/video/og:video): 0      ← אפס רמזים למדיה
script tags: 37                            ← קליפת-JS
```
המדיה לא קיימת ב-HTML — ה-JS בונה את ה-m3u8 (Mux) + token חתום בזמן-ריצה.

### שלב 3 — yt-dlp **+ cookies** (הבדיקה המכריעה)
```
Extracted 3278 cookies from chrome   ← הקוקיז נקראו בהצלחה
[generic] Downloading webpage         ← הוריד את הדף כמשתמש מחובר
ERROR: Unsupported URL                 ← ובכל זאת נכשל
```

**המסקנה:** ב-Maven, cookies לא מספיק. הבעיה היא לא רק login — היא **הרצת JS**. yt-dlp לא מריץ JavaScript, אז גם כמשתמש מחובר ה-HTML שהוא קורא ריק ממדיה.

### למה Playwright הצליח בסשן קודם
| | yt-dlp+cookies | Playwright |
|---|---|---|
| Login (cookies) | ✅ | ✅ |
| **הרצת JS** | ❌ | ✅ |
| Token חתום (נבנה ע"י JS) | ❌ | ✅ |

Playwright רכב על ה-Chrome המחובר, הריץ את ה-JS שבונה את הנגן, ותפס את ה-Mux HLS manifest → playlist הכתוביות → 119 מקטעי VTT → תמלול מלא.

> ⚠️ **הבהרה חשובה:** Playwright **לא תמלל אודיו** ולא הריץ speech-to-text. הוא משך **קובץ-כתוביות (VTT) שכבר היה קיים** ב-Mux ("English CC"). זו הסיבה שזה היה מהיר ומדויק. אם לאתר אין כתוביות מוכנות — אז כן צריך אודיו → Whisper.

---

## בעיות אפשריות ופתירותן (למימוש כמוצר)

| בעיה | חומרה | פתיר? |
|---|---|---|
| כמה וידאו בדף | בינוני | ✅ generic מחזיר `playlist`/`entries` → UI לבחירה או דיפולט "הארוך/הראשי" |
| DRM (Netflix וכו', Widevine) | חוסם | ❌ לא פתיר חוקית — לזהות ולסרב |
| login/paywall | גבוה | ⚠️ פתיר אך רגיש — אחסון/העברת cookies של המשתמש |
| חוקי / ToS / זכויות-יוצרים | גבוה | ⚠️ מדיניות — תנאי-שימוש + אחריות-משתמש + הימנעות מ-DRM |
| SSRF (URL פנימי לשרת) | קריטי | ✅ ולידציה/allow-list, חסימת IP פנימיים |
| עלות/סקייל של דפדפן | בינוני | ✅ תור + מגבלות + cache לפי-URL |
| שבירות (אתר משתנה) | בינוני | ⚠️ yt-dlp מתוחזק בקהילה; סקרייפ ידני = תחזוקה עצמית |
| datacenter-IP נחסם | גבוה | ⚠️ residential/proxy אם מכוונים לאתרים מוגנים |
| קישורים חתומים שפגים | נמוך | ✅ להוריד מיד אחרי תפיסת ה-manifest |

---

## הצעת מימוש מדורגת (MVP → מלא)

1. **שלב 1 (כמעט חינם):** העבר URL-של-דף ל-yt-dlp generic. זהה `Unsupported URL` (= לא נמצא וידאו) ו-`playlist` (= כמה וידאו → בקש בחירה). מכסה את רוב האתרים הציבוריים.
2. **שלב 2:** הוסף `--cookies` לאתרים מאחורי login עם מדיה גלויה (Vimeo וכו').
3. **שלב 3:** Playwright-fallback לדפי-JS + token (Maven-style) — רק כשהשלבים הקודמים נכשלים.
4. **לאורך כל הדרך:** ולידציית-URL (SSRF), זיהוי DRM וסירוב, מדיניות-חוקית/ToS.

**שאלת-מפתח שמכריעה היקף:** אם המיקוד הוא **דפים ציבוריים** → yt-dlp לבד מספיק (זול ופשוט). אם המיקוד הוא **אתרי-קורסים מאחורי login** (כמו Maven) → נכנס כל הכאב (Playwright, cookies, IP, חוקי) וזה כיוון יקר בהרבה.

---

## איך להריץ את ה-POC שוב

```bash
# סט-דגימה ברירת-מחדל
.venv/bin/python scripts/poc_url_extract.py

# על אתרים אמיתיים שלך
.venv/bin/python scripts/poc_url_extract.py "URL1" "URL2" …

# בדיקת cookies על אתר מאחורי login
.venv/bin/yt-dlp --cookies-from-browser chrome --skip-download --list-subs "URL"
```

---

## עדכון 2026-10-01 — דף שכל הסרטונים בו חולקים את כתובת הדף

**מה נשבר:** בדף חי של CNN עם 10 סרטונים, ה-extractor של CNN נותן לכל סרטון
את **כתובת הדף עצמו** כ-`webpage_url`. הבוחר שלח לשרת את אותה כתובת לא משנה מה
נבחר, וההורדה קיבלה את הדף כ-playlist: הורידה את הסרטונים אחד אחרי השני (`noplaylist`
לא עוצר את זה — הוא רלוונטי רק לכתובת שמציינת סרטון *וגם* רשימה), עד שמגבלת 30
הדקות עצרה אותה — 5 סרטונים, 430MB, ושגיאה כללית.

**התיקון:**
- `resolve_video_url` מצרף `item_id` (ה-`id` של הסרטון) לכל מועמד שהכתובת שלו לא
  מזהה אותו לבד — משותפת לכמה מועמדים, או כתובת הדף עצמו.
- הממשק שולח את `item_id`; השרת בוחר את הסרטון לפי ה-id (`match_filter`), **לא לפי
  מיקום** — הדף חי, ובמהלך הבדיקה ירד בו מספר הסרטונים מ-10 ל-9.
- ההורדה מפענחת קודם ומורידה אחר כך (`services/video_selection.py`): דף עם כמה
  סרטונים בלי בחירה נעצר מיד עם `PAGE_HAS_MULTIPLE_VIDEOS`, וסרטון שירד מהדף
  מקבל `VIDEO_NOT_ON_PAGE` — בלי להוריד בייט.
