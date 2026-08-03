"""Logo Manager - Handle logo deduplication and cleanup"""

import hashlib
import logging
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class LogoManager:
    """Manages logo files to prevent duplication and handle cleanup"""

    def __init__(self, assets_folder: str):
        self.assets_folder = assets_folder
        self.logo_prefix = "custom_logo_"

    def get_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def get_file_hash_from_bytes(self, file_bytes: bytes) -> str:
        """Calculate SHA256 hash from bytes"""
        return hashlib.sha256(file_bytes).hexdigest()

    def find_existing_logo(self, file_hash: str) -> str | None:
        """Find if a logo with the same hash already exists"""
        try:
            for filename in os.listdir(self.assets_folder):
                if filename.startswith(self.logo_prefix):
                    file_path = os.path.join(self.assets_folder, filename)
                    if os.path.isfile(file_path):
                        existing_hash = self.get_file_hash(file_path)
                        if existing_hash == file_hash:
                            logger.info(
                                f"Found existing logo with same hash: {filename}"
                            )
                            return file_path
        except Exception as e:
            logger.error(f"Error finding existing logo: {e}")
        return None

    def save_logo(self, file_bytes: bytes, extension: str) -> tuple[str, bool]:
        """
        Save logo file, reusing existing if same content
        Returns: (file_path, is_new)
        """
        # Calculate hash of the new file
        file_hash = self.get_file_hash_from_bytes(file_bytes)

        # Check if we already have this logo
        existing_path = self.find_existing_logo(file_hash)
        if existing_path:
            return existing_path, False

        # Create new file with hash in name for easy identification
        filename = f"{self.logo_prefix}{file_hash[:8]}.{extension}"
        file_path = os.path.join(self.assets_folder, filename)

        # Save the new file
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        logger.info(f"Saved new logo: {filename}")
        return file_path, True

    def cleanup_old_logos(self, keep_hours: int = 24):
        """Remove logo files older than specified hours"""
        try:
            now = datetime.now()
            cutoff_time = now - timedelta(hours=keep_hours)
            removed_count = 0

            for filename in os.listdir(self.assets_folder):
                if filename.startswith(self.logo_prefix):
                    file_path = os.path.join(self.assets_folder, filename)
                    if os.path.isfile(file_path):
                        # Check file modification time
                        file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if file_mtime < cutoff_time:
                            try:
                                os.remove(file_path)
                                removed_count += 1
                                logger.info(f"Removed old logo: {filename}")
                            except Exception as e:
                                logger.error(f"Failed to remove {filename}: {e}")

            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old logo files")

        except Exception as e:
            logger.error(f"Error during logo cleanup: {e}")

    def get_all_logos(self):
        """Get list of all logo files with their info"""
        logos = []
        try:
            for filename in os.listdir(self.assets_folder):
                if filename.startswith(self.logo_prefix):
                    file_path = os.path.join(self.assets_folder, filename)
                    if os.path.isfile(file_path):
                        stat = os.stat(file_path)
                        logos.append(
                            {
                                "filename": filename,
                                "path": file_path,
                                "size": stat.st_size,
                                "modified": datetime.fromtimestamp(stat.st_mtime),
                                "hash": self.get_file_hash(file_path)[:8],
                            }
                        )
        except Exception as e:
            logger.error(f"Error listing logos: {e}")

        return sorted(logos, key=lambda x: x["modified"], reverse=True)
