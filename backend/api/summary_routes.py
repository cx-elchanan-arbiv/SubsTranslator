"""
Summary generation routes for SubsTranslator
Handles AI-powered content summarization using OpenAI GPT
"""

import os

import openai
from celery.result import AsyncResult
from flask import Blueprint, jsonify, request

from api.health_routes import _is_valid_openai_key
from config import get_config
from logging_config import get_logger
from tasks import process_video_task

# Configuration
config = get_config()
logger = get_logger(__name__)

# Create blueprint
summary_bp = Blueprint("summary", __name__)


# Summary prompts in all supported translation languages
SUMMARY_PROMPTS = {
    "he": {
        "system": "אתה עוזר מועיל שמסכם תוכן לפי נושאים עיקריים. תן סיכומים ברורים ומובנים.",
        "user_template": """אנא סכם את התוכן הבא לפי נושאים עיקריים.

התוכן:
{text}

אנא צור סיכום מובנה שמחלק את התוכן לפי נושאים עיקריים, עם נקודות מפתח תחת כל נושא.
השתמש בפורמט markdown עם כותרות וסעיפים.""",
    },
    "en": {
        "system": "You are a helpful assistant that summarizes content by main topics. Provide clear and understandable summaries.",
        "user_template": """Please summarize the following content by main topics.

Content:
{text}

Please create a structured summary that divides the content by main topics, with key points under each topic.
Use markdown format with headers and bullet points.""",
    },
    "es": {
        "system": "Eres un asistente útil que resume contenido por temas principales. Proporciona resúmenes claros y comprensibles.",
        "user_template": """Por favor resume el siguiente contenido por temas principales.

Contenido:
{text}

Por favor crea un resumen estructurado que divida el contenido por temas principales, con puntos clave bajo cada tema.
Usa formato markdown con encabezados y viñetas.""",
    },
    "ar": {
        "system": "أنت مساعد مفيد يلخص المحتوى حسب المواضيع الرئيسية. قدم ملخصات واضحة ومفهومة.",
        "user_template": """يرجى تلخيص المحتوى التالي حسب المواضيع الرئيسية.

المحتوى:
{text}

يرجى إنشاء ملخص منظم يقسم المحتوى حسب المواضيع الرئيسية، مع النقاط الرئيسية تحت كل موضوع.
استخدم تنسيق markdown مع العناوين والنقاط.""",
    },
    "fr": {
        "system": "Vous êtes un assistant utile qui résume le contenu par thèmes principaux. Fournissez des résumés clairs et compréhensibles.",
        "user_template": """Veuillez résumer le contenu suivant par thèmes principaux.

Contenu:
{text}

Veuillez créer un résumé structuré qui divise le contenu par thèmes principaux, avec des points clés sous chaque thème.
Utilisez le format markdown avec des en-têtes et des puces.""",
    },
    "de": {
        "system": "Sie sind ein hilfreicher Assistent, der Inhalte nach Hauptthemen zusammenfasst. Geben Sie klare und verständliche Zusammenfassungen.",
        "user_template": """Bitte fassen Sie den folgenden Inhalt nach Hauptthemen zusammen.

Inhalt:
{text}

Bitte erstellen Sie eine strukturierte Zusammenfassung, die den Inhalt nach Hauptthemen unterteilt, mit wichtigen Punkten unter jedem Thema.
Verwenden Sie das Markdown-Format mit Überschriften und Aufzählungszeichen.""",
    },
    "it": {
        "system": "Sei un assistente utile che riassume i contenuti per argomenti principali. Fornisci riassunti chiari e comprensibili.",
        "user_template": """Per favore riassumi il seguente contenuto per argomenti principali.

Contenuto:
{text}

Per favore crea un riassunto strutturato che divide il contenuto per argomenti principali, con punti chiave sotto ogni argomento.
Usa il formato markdown con intestazioni e elenchi puntati.""",
    },
    "pt": {
        "system": "Você é um assistente útil que resume conteúdo por tópicos principais. Forneça resumos claros e compreensíveis.",
        "user_template": """Por favor, resuma o seguinte conteúdo por tópicos principais.

Conteúdo:
{text}

Por favor, crie um resumo estruturado que divida o conteúdo por tópicos principais, com pontos-chave sob cada tópico.
Use o formato markdown com cabeçalhos e marcadores.""",
    },
    "ru": {
        "system": "Вы полезный помощник, который резюмирует содержание по основным темам. Предоставляйте четкие и понятные резюме.",
        "user_template": """Пожалуйста, резюмируйте следующий контент по основным темам.

Содержание:
{text}

Пожалуйста, создайте структурированное резюме, которое разделяет контент по основным темам, с ключевыми моментами под каждой темой.
Используйте формат markdown с заголовками и маркерами.""",
    },
    "ja": {
        "system": "あなたは主要なトピックごとにコンテンツを要約する役立つアシスタントです。明確で理解しやすい要約を提供してください。",
        "user_template": """以下のコンテンツを主要なトピックごとに要約してください。

コンテンツ:
{text}

主要なトピックごとにコンテンツを分割し、各トピックの下に重要なポイントを含む構造化された要約を作成してください。
見出しと箇条書きを含むマークダウン形式を使用してください。""",
    },
    "ko": {
        "system": "당신은 주요 주제별로 콘텐츠를 요약하는 유용한 어시스턴트입니다. 명확하고 이해하기 쉬운 요약을 제공하세요.",
        "user_template": """다음 콘텐츠를 주요 주제별로 요약해주세요.

콘텐츠:
{text}

주요 주제별로 콘텐츠를 나누고 각 주제 아래에 핵심 포인트를 포함한 구조화된 요약을 작성해주세요.
제목과 글머리 기호가 포함된 마크다운 형식을 사용하세요.""",
    },
    "zh": {
        "system": "您是一个有用的助手，按主要主题总结内容。提供清晰易懂的摘要。",
        "user_template": """请按主要主题总结以下内容。

内容：
{text}

请创建一个结构化的摘要，按主要主题划分内容，每个主题下包含关键要点。
使用带有标题和项目符号的markdown格式。""",
    },
    "tr": {
        "system": "Ana konulara göre içeriği özetleyen yardımcı bir asistansınız. Açık ve anlaşılır özetler sağlayın.",
        "user_template": """Lütfen aşağıdaki içeriği ana konulara göre özetleyin.

İçerik:
{text}

Lütfen içeriği ana konulara göre bölen, her konunun altında önemli noktaların olduğu yapılandırılmış bir özet oluşturun.
Başlıklar ve madde işaretleri içeren markdown formatını kullanın.""",
    },
}


def _extract_text_from_srt(filepath: str) -> str:
    """Extract text content from SRT file, removing timestamps and numbering"""
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()

        # Parse SRT format
        # SRT format: number, timestamp, text, blank line
        lines = content.split("\n")
        text_lines = []

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines and numbers
            if not line or line.isdigit():
                i += 1
                continue

            # Skip timestamp lines (format: 00:00:00,000 --> 00:00:00,000)
            if "-->" in line:
                i += 1
                continue

            # This is actual subtitle text
            text_lines.append(line)
            i += 1

        return " ".join(text_lines)

    except Exception as e:
        logger.error(f"Failed to extract text from SRT: {e}")
        raise


def _generate_summary_with_openai(
    text: str, lang: str = "he", custom_prompt: str = None
) -> str:
    """Generate topic-based summary using OpenAI GPT in the specified language

    Args:
        text: The subtitle text to summarize
        lang: Language code for the summary
        custom_prompt: Optional custom instructions from user. If provided, this overrides the default prompt.
    """
    try:
        import httpx

        http_client = httpx.Client(timeout=60.0)
        client = openai.OpenAI(api_key=config.OPENAI_API_KEY, http_client=http_client)

        if custom_prompt:
            # User provided custom instructions - use them directly
            # Keep the same system message for consistency
            prompts = SUMMARY_PROMPTS.get(lang, SUMMARY_PROMPTS["he"])
            user_prompt = (
                f"{custom_prompt}\n\nהתוכן:\n{text}"
                if lang in ["he", "ar"]
                else f"{custom_prompt}\n\nContent:\n{text}"
            )

            logger.info(f"Using custom prompt (length: {len(custom_prompt)} chars)")
        else:
            # Use default prompts for the specified language
            prompts = SUMMARY_PROMPTS.get(lang, SUMMARY_PROMPTS["he"])
            user_prompt = prompts["user_template"].format(text=text)

            logger.info(f"Using default prompt for {lang}")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        prompts["system"] if not custom_prompt else prompts["system"]
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            max_tokens=2000,
            timeout=60,
        )

        summary = response.choices[0].message.content.strip()
        logger.info(
            f"Summary generated successfully in {lang} ({len(summary)} characters)"
        )

        return summary

    except Exception as e:
        logger.error(f"OpenAI summary generation failed: {e}")
        raise


@summary_bp.route("/api/summaries", methods=["POST"])
def create_summary():
    """Create a summary of translated subtitles by topics using GPT (New secure endpoint)"""
    try:
        data = request.get_json()

        # Validate request
        if not data:
            return jsonify({"error": "No data provided"}), 400

        task_id = data.get("task_id")
        summary_lang = data.get("summary_lang")
        custom_prompt = data.get("custom_prompt")  # Optional custom prompt from user

        if not task_id:
            return jsonify({"error": "task_id is required"}), 400

        if not summary_lang:
            return jsonify({"error": "summary_lang is required"}), 400

        # Validate language code against all supported translation languages
        from shared_config import TARGET_LANGUAGES

        if summary_lang not in TARGET_LANGUAGES:
            return (
                jsonify(
                    {
                        "error": f"Invalid summary_lang. Supported: {', '.join(TARGET_LANGUAGES)}"
                    }
                ),
                400,
            )

        # Validate custom prompt length if provided
        if custom_prompt and len(custom_prompt) > 1500:
            return (
                jsonify(
                    {
                        "error": f"custom_prompt is too long ({len(custom_prompt)} characters). Maximum is 1500 characters."
                    }
                ),
                400,
            )

        # Check if OpenAI API key is configured
        if not _is_valid_openai_key(config.OPENAI_API_KEY):
            return (
                jsonify(
                    {
                        "error": "OpenAI API key is not configured. This feature requires OpenAI."
                    }
                ),
                503,
            )

        # Retrieve task result to get filename
        task_result = AsyncResult(task_id, app=process_video_task.app)

        if task_result.state != "SUCCESS":
            return (
                jsonify(
                    {
                        "error": f"Task is not completed yet. Current state: {task_result.state}"
                    }
                ),
                400,
            )

        result = task_result.result
        if not result or not isinstance(result, dict):
            return jsonify({"error": "Invalid task result"}), 500

        # Handle nested result structure (same as /status endpoint)
        # Result can be: {"status": "SUCCESS", "result": {...}} or just {...}
        actual_result = result.get("result", result) if "result" in result else result

        # Extract translated SRT filename from result
        files = actual_result.get("files", {})
        translated_srt = files.get("translated_srt")

        if not translated_srt:
            return (
                jsonify({"error": "No translated subtitles found for this task"}),
                404,
            )

        # Security: prevent path traversal
        safe_dir = os.path.abspath(config.DOWNLOADS_FOLDER)
        requested_path = os.path.abspath(os.path.join(safe_dir, translated_srt))

        if not requested_path.startswith(safe_dir):
            return jsonify({"error": "Forbidden"}), 403

        if not os.path.exists(requested_path):
            return jsonify({"error": "File not found"}), 404

        # Read SRT file and extract text content
        logger.info(f"Reading subtitles from {translated_srt} for task {task_id}")
        subtitle_text = _extract_text_from_srt(requested_path)

        if not subtitle_text:
            return jsonify({"error": "No text content found in subtitle file"}), 400

        # Summarize using OpenAI in the requested language
        if custom_prompt:
            logger.info(
                f"Generating custom summary in {summary_lang} for task {task_id}"
            )
            summary = _generate_summary_with_openai(
                subtitle_text, lang=summary_lang, custom_prompt=custom_prompt
            )
        else:
            logger.info(
                f"Generating default summary in {summary_lang} for task {task_id}"
            )
            summary = _generate_summary_with_openai(subtitle_text, lang=summary_lang)

        return (
            jsonify(
                {
                    "success": True,
                    "summary": summary,
                    "task_id": task_id,
                    "summary_lang": summary_lang,
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return jsonify({"error": str(e)}), 500


@summary_bp.route("/summarize/<filename>", methods=["GET"])
def summarize_subtitles(filename):
    """[DEPRECATED] Summarize translated subtitles by topics using GPT - Use POST /api/summaries instead"""
    try:
        # Check if OpenAI API key is configured
        if not _is_valid_openai_key(config.OPENAI_API_KEY):
            return (
                jsonify(
                    {
                        "error": "OpenAI API key is not configured. This feature requires OpenAI."
                    }
                ),
                503,
            )

        # Security: prevent path traversal
        safe_dir = os.path.abspath(config.DOWNLOADS_FOLDER)
        requested_path = os.path.abspath(os.path.join(safe_dir, filename))

        if not requested_path.startswith(safe_dir):
            return jsonify({"error": "Forbidden"}), 403

        if not os.path.exists(requested_path):
            return jsonify({"error": "File not found"}), 404

        # Read SRT file and extract text content
        logger.info(f"Reading subtitles from {filename}")
        subtitle_text = _extract_text_from_srt(requested_path)

        if not subtitle_text:
            return jsonify({"error": "No text content found in subtitle file"}), 400

        # Summarize using OpenAI (default to Hebrew for backward compatibility)
        logger.info(f"Generating summary for {filename}")
        summary = _generate_summary_with_openai(subtitle_text, lang="he")

        return jsonify({"success": True, "summary": summary, "filename": filename}), 200

    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return jsonify({"error": str(e)}), 500
