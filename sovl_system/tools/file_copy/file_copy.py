import shutil
import os
from sovl_logger import Logger
from sovl_error import ErrorManager

def file_copy(source_path: str, dest_path: str, overwrite: bool = False) -> str:
    logger = Logger.get_instance()
    error_manager = ErrorManager.get_instance()
    try:
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")
        if os.path.exists(dest_path) and not overwrite:
            raise FileExistsError(f"Destination file exists: {dest_path}")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(source_path, dest_path)
        logger.info(f"File copied from {source_path} to {dest_path}")
        return f"File copied to {dest_path}"
    except Exception as e:
        logger.error(f"Error copying file from {source_path} to {dest_path}: {e}", exc_info=True)
        error_manager.record_error(
            error=e,
            error_type="file_copy_error",
            context={"source_path": source_path, "dest_path": dest_path, "overwrite": overwrite}
        )
        raise

ACTIVATION_PHRASES = ["copy file", "duplicate file", "move file"]