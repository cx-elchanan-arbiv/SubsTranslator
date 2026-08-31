"""
Video editing routes for SubsTranslator
Handles video cutting, merging, subtitle embedding, and watermark addition
"""

import os
import uuid

from flask import Blueprint, jsonify, request, send_file, session

from config import get_config
from logging_config import get_logger
from logo_manager import LogoManager
from utils.file_utils import clean_filename, safe_int
from utils.video_utils import (
    add_watermark_to_video,
    cut_video_ffmpeg,
    embed_subtitles_ffmpeg,
    merge_videos_ffmpeg,
    parse_text_to_srt,
)

# Configuration
config = get_config()
logger = get_logger(__name__)

# Initialize logo manager
logo_manager = LogoManager(config.ASSETS_FOLDER)

# Create blueprint
editing_bp = Blueprint("editing", __name__)


@editing_bp.route("/cut-video", methods=["POST"])
def cut_video():
    """Cut a video from start_time to end_time with ultra-precise FFmpeg cutting."""
    try:
        # Check if video file was uploaded
        if "video" not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files["video"]
        if video_file.filename == "":
            return jsonify({"error": "No video file selected"}), 400

        # Get time parameters
        start_time = request.form.get("start_time", "00:00:00")
        end_time = request.form.get("end_time", "00:01:00")

        logger.info(
            f"Cutting video: {video_file.filename} from {start_time} to {end_time}"
        )

        # Save uploaded video
        video_filename = clean_filename(video_file.filename)
        input_path = os.path.join(
            config.UPLOAD_FOLDER, f"cut_input_{uuid.uuid4()}_{video_filename}"
        )
        video_file.save(input_path)

        # Prepare output path
        output_filename = f"cut_{start_time.replace(':', '')}_{end_time.replace(':', '')}_{video_filename}"
        output_path = os.path.join(config.DOWNLOADS_FOLDER, output_filename)

        # Cut the video
        success = cut_video_ffmpeg(input_path, output_path, start_time, end_time)

        if not success:
            # Clean up
            if os.path.exists(input_path):
                os.remove(input_path)
            return (
                jsonify(
                    {
                        "error": "Failed to cut video. Please check the time format and try again."
                    }
                ),
                500,
            )

        # Clean up input file
        if os.path.exists(input_path):
            os.remove(input_path)

        # Return the cut video
        logger.info(f"Video cut successfully: {output_filename}")
        return send_file(
            output_path,
            as_attachment=True,
            download_name=output_filename,
            mimetype="video/mp4",
        )

    except Exception as e:
        logger.error(f"Video cutting failed: {e}")
        # Clean up on error
        if "input_path" in locals() and os.path.exists(input_path):
            os.remove(input_path)
        if "output_path" in locals() and os.path.exists(output_path):
            os.remove(output_path)
        return jsonify({"error": str(e)}), 500


@editing_bp.route("/embed-subtitles", methods=["POST"])
def embed_subtitles():
    """Embed subtitles into video with optional watermark."""
    try:
        # Check if video file was uploaded
        if "video" not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files["video"]
        if video_file.filename == "":
            return jsonify({"error": "No video file selected"}), 400

        # Get subtitles (file or text)
        srt_file = request.files.get("srt_file")
        srt_text = request.form.get("srt_text", "")

        if not srt_file and not srt_text:
            return jsonify({"error": "Please provide subtitles (file or text)"}), 400

        # Get logo options
        include_logo = request.form.get("include_logo", "false").lower() == "true"
        logo_position = request.form.get("logo_position", "bottom-right")
        logo_size = request.form.get("logo_size", "medium")
        logo_opacity, opacity_error = safe_int(
            request.form.get("logo_opacity"), 40, 0, 100
        )
        if opacity_error:
            return jsonify({"error": f"Invalid logo_opacity: {opacity_error}"}), 400

        logger.info(f"Embedding subtitles into: {video_file.filename}")

        # Save uploaded video
        video_filename = clean_filename(video_file.filename)
        input_video_path = os.path.join(
            config.UPLOAD_FOLDER, f"embed_input_{uuid.uuid4()}_{video_filename}"
        )
        video_file.save(input_video_path)

        # Handle subtitles
        if srt_file:
            # Save SRT file
            srt_filename = clean_filename(srt_file.filename)
            srt_path = os.path.join(
                config.UPLOAD_FOLDER, f"srt_{uuid.uuid4()}_{srt_filename}"
            )
            srt_file.save(srt_path)
        else:
            # Parse text to SRT
            srt_path = os.path.join(config.UPLOAD_FOLDER, f"srt_{uuid.uuid4()}.srt")
            success = parse_text_to_srt(srt_text, srt_path)
            if not success:
                if os.path.exists(input_video_path):
                    os.remove(input_video_path)
                return (
                    jsonify(
                        {
                            "error": "Failed to parse subtitles text. Please check the format."
                        }
                    ),
                    400,
                )

        # Embed subtitles
        temp_output_path = os.path.join(
            config.DOWNLOADS_FOLDER, f"with_subs_{uuid.uuid4()}_{video_filename}"
        )
        success = embed_subtitles_ffmpeg(input_video_path, srt_path, temp_output_path)

        if not success:
            # Clean up
            for path in [input_video_path, srt_path]:
                if os.path.exists(path):
                    os.remove(path)
            return jsonify({"error": "Failed to embed subtitles"}), 500

        # Add watermark if requested
        if include_logo:
            # The user's uploaded logo is persisted by the watermark upload routes
            # (api/video_routes.py) into the Flask session under 'custom_logo_path'
            # — that is the app-wide contract for "this user's logo". Fall back to
            # the configured default watermark asset.
            logo_path = session.get("custom_logo_path")
            if not logo_path or not os.path.exists(logo_path):
                logo_path = config.get_watermark_path("default")

            if os.path.exists(logo_path):
                final_output_path = os.path.join(
                    config.DOWNLOADS_FOLDER, f"final_{uuid.uuid4()}_{video_filename}"
                )
                success = add_watermark_to_video(
                    temp_output_path,
                    final_output_path,
                    logo_path,
                    logo_position,
                    logo_size,
                    logo_opacity,
                )

                if success:
                    # Remove temp file
                    if os.path.exists(temp_output_path):
                        os.remove(temp_output_path)
                    output_path = final_output_path
                else:
                    logger.warning(
                        "Failed to add watermark, returning video with subtitles only"
                    )
                    output_path = temp_output_path
            else:
                logger.warning("Logo not found, returning video with subtitles only")
                output_path = temp_output_path
        else:
            output_path = temp_output_path

        # Clean up input files
        for path in [input_video_path, srt_path]:
            if os.path.exists(path):
                os.remove(path)

        # Return the video
        output_filename = f"video_with_subtitles_{video_filename}"
        logger.info(f"Subtitles embedded successfully: {output_filename}")
        return send_file(
            output_path,
            as_attachment=True,
            download_name=output_filename,
            mimetype="video/mp4",
        )

    except Exception as e:
        logger.error(f"Embed subtitles failed: {e}")
        # Clean up on error
        for var in [
            "input_video_path",
            "srt_path",
            "temp_output_path",
            "final_output_path",
            "output_path",
        ]:
            if var in locals():
                path = locals()[var]
                if path and os.path.exists(path):
                    os.remove(path)
        return jsonify({"error": str(e)}), 500


@editing_bp.route("/merge-videos", methods=["POST"])
def merge_videos():
    """Merge two videos with automatic resolution handling."""
    try:
        # Get uploaded video files
        video1_file = request.files.get("video1")
        video2_file = request.files.get("video2")

        if not video1_file or not video2_file:
            return jsonify({"error": "Both video files are required"}), 400

        # Generate unique filenames
        video1_id = str(uuid.uuid4())
        video2_id = str(uuid.uuid4())
        output_id = str(uuid.uuid4())

        # Get file extensions
        video1_ext = os.path.splitext(video1_file.filename)[1] or ".mp4"
        video2_ext = os.path.splitext(video2_file.filename)[1] or ".mp4"

        # Save uploaded files
        video1_path = os.path.join(
            config.UPLOAD_FOLDER, f"video1_{video1_id}{video1_ext}"
        )
        video2_path = os.path.join(
            config.UPLOAD_FOLDER, f"video2_{video2_id}{video2_ext}"
        )
        output_path = os.path.join(config.DOWNLOADS_FOLDER, f"merged_{output_id}.mp4")

        video1_file.save(video1_path)
        video2_file.save(video2_path)

        logger.info(f"Merging videos: {video1_path} + {video2_path} -> {output_path}")

        # Merge videos using FFmpeg
        success = merge_videos_ffmpeg(video1_path, video2_path, output_path)

        if not success:
            # Cleanup on failure
            if os.path.exists(video1_path):
                os.remove(video1_path)
            if os.path.exists(video2_path):
                os.remove(video2_path)
            if os.path.exists(output_path):
                os.remove(output_path)
            return jsonify({"error": "Failed to merge videos"}), 500

        # Return merged video
        output_filename = f"merged_{video1_file.filename.split('.')[0]}_{video2_file.filename.split('.')[0]}.mp4"

        logger.info(f"Videos merged successfully: {output_path}")

        # Cleanup input files after sending (output will be cleaned by periodic cleanup)
        try:
            if os.path.exists(video1_path):
                os.remove(video1_path)
            if os.path.exists(video2_path):
                os.remove(video2_path)
        except Exception as cleanup_error:
            logger.warning(f"Cleanup error: {cleanup_error}")

        return send_file(
            output_path,
            as_attachment=True,
            download_name=output_filename,
            mimetype="video/mp4",
        )

    except Exception as e:
        logger.error(f"Error in merge_videos endpoint: {str(e)}", exc_info=True)

        # Cleanup on error
        for var in ["video1_path", "video2_path", "output_path"]:
            if var in locals():
                path = locals()[var]
                if path and os.path.exists(path):
                    os.remove(path)
        return jsonify({"error": str(e)}), 500


@editing_bp.route("/extract-audio", methods=["POST"])
def extract_audio():
    """Extract audio from a video file and return it as MP3."""
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files["video"]
        if video_file.filename == "":
            return jsonify({"error": "No video file selected"}), 400

        audio_format = request.form.get("format", "mp3")
        if audio_format not in ("mp3", "wav"):
            audio_format = "mp3"

        logger.info(f"Extracting audio from: {video_file.filename} as {audio_format}")

        # Save uploaded video
        video_filename = clean_filename(video_file.filename)
        input_path = os.path.join(
            config.UPLOAD_FOLDER, f"audio_input_{uuid.uuid4()}_{video_filename}"
        )
        video_file.save(input_path)

        # Prepare output path
        base_name = os.path.splitext(video_filename)[0]
        output_filename = f"{base_name}_audio.{audio_format}"
        output_path = os.path.join(
            config.DOWNLOADS_FOLDER, f"audio_{uuid.uuid4()}_{output_filename}"
        )

        # Extract audio with FFmpeg
        import subprocess

        if audio_format == "mp3":
            cmd = [
                "ffmpeg",
                "-i",
                input_path,
                "-vn",
                "-acodec",
                "libmp3lame",
                "-ab",
                "192k",
                "-y",
                output_path,
            ]
            mimetype = "audio/mpeg"
        else:
            cmd = [
                "ffmpeg",
                "-i",
                input_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "44100",
                "-y",
                output_path,
            ]
            mimetype = "audio/wav"

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            logger.error(f"FFmpeg audio extraction failed: {result.stderr}")
            if os.path.exists(input_path):
                os.remove(input_path)
            return jsonify({"error": "Failed to extract audio from video"}), 500

        # Clean up input file
        if os.path.exists(input_path):
            os.remove(input_path)

        logger.info(f"Audio extracted successfully: {output_filename}")
        return send_file(
            output_path,
            as_attachment=True,
            download_name=output_filename,
            mimetype=mimetype,
        )

    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        if "input_path" in locals() and os.path.exists(input_path):
            os.remove(input_path)
        if "output_path" in locals() and os.path.exists(output_path):
            os.remove(output_path)
        return jsonify({"error": str(e)}), 500


@editing_bp.route("/add-logo-to-video", methods=["POST"])
def add_logo_to_video():
    """Add a logo/watermark to a video without transcription or translation."""
    try:
        # Get uploaded files
        video_file = request.files.get("video")
        logo_file = request.files.get("logo")

        if not video_file:
            return jsonify({"error": "Video file is required"}), 400

        if not logo_file:
            return jsonify({"error": "Logo file is required"}), 400

        # Get watermark settings
        position = request.form.get("position", "top-right")
        size = request.form.get("size", "medium")
        opacity, opacity_error = safe_int(request.form.get("opacity"), 40, 0, 100)
        if opacity_error:
            return jsonify({"error": f"Invalid opacity: {opacity_error}"}), 400

        logger.info(
            f"Adding logo to video: {video_file.filename} with position={position}, size={size}, opacity={opacity}"
        )

        # Generate unique IDs
        video_id = str(uuid.uuid4())
        logo_id = str(uuid.uuid4())
        output_id = str(uuid.uuid4())

        # Get file extensions
        video_ext = os.path.splitext(video_file.filename)[1] or ".mp4"
        logo_ext = os.path.splitext(logo_file.filename)[1] or ".png"

        # Save uploaded files
        video_path = os.path.join(config.UPLOAD_FOLDER, f"video_{video_id}{video_ext}")
        logo_path = os.path.join(config.UPLOAD_FOLDER, f"logo_{logo_id}{logo_ext}")
        output_path = os.path.join(
            config.DOWNLOADS_FOLDER, f"with_logo_{output_id}.mp4"
        )

        video_file.save(video_path)
        logo_file.save(logo_path)

        # Add watermark to video
        success = add_watermark_to_video(
            video_path,
            output_path,
            logo_path,
            position=position,
            size=size,
            opacity=opacity,
        )

        if not success:
            # Cleanup on failure
            for path in [video_path, logo_path, output_path]:
                if os.path.exists(path):
                    os.remove(path)
            return jsonify({"error": "Failed to add logo to video"}), 500

        # Prepare output filename
        video_basename = os.path.splitext(video_file.filename)[0]
        output_filename = f"{video_basename}_with_logo.mp4"

        logger.info(f"Logo added successfully: {output_path}")

        # Cleanup input files
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(logo_path):
                os.remove(logo_path)
        except Exception as cleanup_error:
            logger.warning(f"Cleanup error: {cleanup_error}")

        # Return the video with logo
        return send_file(
            output_path,
            as_attachment=True,
            download_name=output_filename,
            mimetype="video/mp4",
        )

    except Exception as e:
        logger.error(f"Error in add_logo_to_video endpoint: {str(e)}", exc_info=True)

        # Cleanup on error
        for var in ["video_path", "logo_path", "output_path"]:
            if var in locals():
                path = locals()[var]
                if path and os.path.exists(path):
                    os.remove(path)

        return jsonify({"error": str(e)}), 500
