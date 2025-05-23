import pyperclip
from sovl_logger import Logger
from sovl_error import ErrorManager

def clipboard_copy(text: str) -> str:
    logger = Logger.get_instance()
    error_manager = ErrorManager.get_instance()
    try:
        pyperclip.copy(text)
        logger.info(f"Copied text to clipboard: {text[:50]}...")
        return "Text copied to clipboard"
    except Exception as e:
        logger.error(f"Error copying to clipboard: {e}", exc_info=True)
        error_manager.record_error(
            error=e,
            error_type="clipboard_copy_error",
            context={"text": text[:50]}
        )
        raise

ACTIVATION_PHRASES = ["copy to clipboard", "save to clipboard", "clipboard set"]