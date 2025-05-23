import os
from sovl_logger import Logger
from sovl_error import ErrorManager

def list_directory(path: str = ".", include_hidden: bool = False) -> list:
    logger = Logger.get_instance()
    error_manager = ErrorManager.get_instance()
    try:
        entries = os.listdir(path)
        if not include_hidden:
            entries = [e for e in entries if not e.startswith('.')]
        logger.info(f"Listed directory: {path}, include_hidden={include_hidden}")
        return entries
    except Exception as e:
        logger.error(f"Error listing directory {path}: {e}", exc_info=True)
        error_manager.record_error(
            error=e,
            error_type="list_directory_error",
            context={"path": path, "include_hidden": include_hidden}
        )
        raise

ACTIVATION_PHRASES = ["list files", "show directory", "dir contents"]