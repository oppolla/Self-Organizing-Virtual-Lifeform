import os
from sovl_logger import Logger
from sovl_error import ErrorManager

def file_write(file_path: str, content: str, mode: str = "write", encoding: str = "utf-8") -> str:
    logger = Logger.get_instance()
    error_manager = ErrorManager.get_instance()
    try:
        file_mode = 'w' if mode == "write" else 'a'
        with open(file_path, file_mode, encoding=encoding) as f:
            f.write(content)
        logger.info(f"Successfully wrote to file: {file_path} (mode: {mode})")
        return f"Content written to {file_path}"
    except Exception as e:
        logger.error(f"Error writing to file {file_path}: {e}", exc_info=True)
        error_manager.record_error(
            error=e,
            error_type="file_write_error",
            context={"file_path": file_path, "mode": mode, "encoding": encoding}
        )
        raise

ACTIVATION_PHRASES = ["write file", "save to file", "append to file"]