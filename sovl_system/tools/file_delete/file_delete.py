import os
from sovl_logger import Logger
from sovl_error import ErrorManager

def file_delete(file_path: str) -> str:
    logger = Logger.get_instance()
    error_manager = ErrorManager.get_instance()
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        os.remove(file_path)
        logger.info(f"File deleted: {file_path}")
        return f"File {file_path} deleted"
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}", exc_info=True)
        error_manager.record_error(
            error=e,
            error_type="file_delete_error",
            context={"file_path": file_path}
        )
        raise

ACTIVATION_PHRASES = ["delete file", "remove file", "erase file"]