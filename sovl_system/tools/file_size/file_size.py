import os
from sovl_logger import Logger
from sovl_error import ErrorManager

def file_size(file_path: str, human_readable: bool = False) -> str:
    logger = Logger.get_instance()
    error_manager = ErrorManager.get_instance()
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        size = os.path.getsize(file_path)
        if human_readable:
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024:
                    break
                size /= 1024
            result = f"{size:.2f} {unit}"
        else:
            result = str(size)
        logger.info(f"File size for {file_path}: {result}")
        return result
    except Exception as e:
        logger.error(f"Error getting file size for {file_path}: {e}", exc_info=True)
        error_manager.record_error(
            error=e,
            error_type="file_size_error",
            context={"file_path": file_path, "human_readable": human_readable}
        )
        raise

ACTIVATION_PHRASES = ["file size", "check file size", "get file size"]