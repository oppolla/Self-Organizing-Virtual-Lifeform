import os
from sovl_logger import Logger
from sovl_error import ErrorManager

def create_directory(path: str, parents: bool = True) -> str:
    logger = Logger.get_instance()
    error_manager = ErrorManager.get_instance()
    try:
        os.makedirs(path, exist_ok=True) if parents else os.mkdir(path)
        logger.info(f"Directory created: {path}")
        return f"Directory created: {path}"
    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}", exc_info=True)
        error_manager.record_error(
            error=e,
            error_type="create_directory_error",
            context={"path": path, "parents": parents}
        )
        raise

ACTIVATION_PHRASES = ["create folder", "make directory", "new folder"]