import sys
import subprocess
from sovl_logger import Logger
from sovl_error import ErrorManager

def clipboard_paste() -> str:
    logger = Logger.get_instance()
    error_manager = ErrorManager.get_instance()
    try:
        if sys.platform == "darwin":  # macOS
            p = subprocess.Popen(['pbpaste'], stdout=subprocess.PIPE)
            text, _ = p.communicate()
            text = text.decode('utf-8')
        elif sys.platform == "win32":  # Windows
            # Use PowerShell's Get-Clipboard
            p = subprocess.Popen(['powershell', '-command', 'Get-Clipboard'], stdout=subprocess.PIPE)
            text, _ = p.communicate()
            text = text.decode('utf-8')
        else:
            raise NotImplementedError("Clipboard paste is only supported on macOS and Windows in this implementation.")
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