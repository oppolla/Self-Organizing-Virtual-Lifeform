from datetime import datetime
from sovl_logger import Logger
from sovl_error import ErrorManager

def current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    logger = Logger.get_instance()
    error_manager = ErrorManager.get_instance()
    try:
        result = datetime.now().strftime(format)
        logger.info(f"Retrieved current time: {result}")
        return result
    except Exception as e:
        logger.error(f"Error retrieving current time: {e}", exc_info=True)
        error_manager.record_error(
            error=e,
            error_type="current_time_error",
            context={"format": format}
        )
        raise

ACTIVATION_PHRASES = ["get time", "current time", "what time is it"]