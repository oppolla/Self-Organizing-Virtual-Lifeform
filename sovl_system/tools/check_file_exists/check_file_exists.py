import os
from sovl_logger import Logger
from sovl_error import ErrorManager

def check_file_exists(path: str, check_type: str = "any") -> bool:
    logger = Logger.get_instance()
    error_manager = ErrorManager.get_instance()
    try:
        exists = os.path.exists(path)
        if not exists:
            logger.info(f"Path does not exist: {path}")
            return False
        if check_type == "file":
            result = os.path.isfile(path)
        elif check_type == "directory":
            result = os.path.isdir(path)
        elif check_type == "any":
            result = True
        else:
            raise ValueError(f"Invalid check_type: {check_type}")
        logger.info(f"Checked path {path} (type: {check_type}): {result}")
        return result
    except Exception as e:
        logger.error(f"Error checking path {path}: {e}", exc_info=True)
        error_manager.record_error(
            error=e,
            error_type="check_file_exists_error",
            context={"path": path, "check_type": check_type}
        )
        raise

ACTIVATION_PHRASES = ["check file", "file exists", "verify path"]