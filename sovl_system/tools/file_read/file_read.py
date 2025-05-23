import os
from sovl_logger import Logger
from sovl_error import ErrorManager

def file_read(file_path: str, encoding: str = "utf-8") -> str:
    logger = Logger.get_instance()
    error_manager = ErrorManager.get_instance()
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            error_manager.record_error(
                error=FileNotFoundError(f"File not found: {file_path}"),
                error_type="file_read_not_found",
                context={"file_path": file_path}
            )
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        logger.info(f"Successfully read file: {file_path}")
        return content
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
        error_manager.record_error(
            error=e,
            error_type="file_read_error",
            context={"file_path": file_path, "encoding": encoding}
        )
        raise

ACTIVATION_PHRASES = ["read file", "open file", "get file contents"]