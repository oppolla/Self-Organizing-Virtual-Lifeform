import pyperclip
from sovl_logger import Logger
from sovl_error import ErrorManager

def clipboard_paste() -> str:
    logger = Logger.get_instance()
    error_manager = ErrorManager.get_instance()
    try:
        text = pyperclip.paste()
        logger.info(f"Retrieved text from clipboard: {text[:50]}...")
        return text
    except Exception as e:
        logger.error(f"Error pasting from clipboard: {e}", exc_info=True)
        error_manager.record_error(
            error=e,
            error_type="clipboard_paste_error",
            context={}
        )
        raise

ACTIVATION_PHRASES = ["paste from clipboard", "get clipboard", "read clipboard"]