import re
from sovl_logger import Logger
from sovl_error import ErrorManager

def string_replace(text: str, pattern: str, replacement: str, is_regex: bool = False) -> str:
    logger = Logger.get_instance()
    error_manager = ErrorManager.get_instance()
    try:
        if is_regex:
            result = re.sub(pattern, replacement, text)
        else:
            result = text.replace(pattern, replacement)
        logger.info(f"String replacement completed: pattern='{pattern}', replacement='{replacement}', is_regex={is_regex}")
        return result
    except Exception as e:
        logger.error(f"Error in string replacement: {e}", exc_info=True)
        error_manager.record_error(
            error=e,
            error_type="string_replace_error",
            context={"text": text, "pattern": pattern, "replacement": replacement, "is_regex": is_regex}
        )
        raise

ACTIVATION_PHRASES = ["replace text", "string replace", "substitute text"]