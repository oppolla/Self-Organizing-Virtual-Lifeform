import json
from sovl_logger import Logger
from sovl_error import ErrorManager

def json_parse(json_string: str) -> dict:
    logger = Logger.get_instance()
    error_manager = ErrorManager.get_instance()
    try:
        result = json.loads(json_string)
        logger.info("JSON parsed successfully")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON: {e}", exc_info=True)
        error_manager.record_error(
            error=e,
            error_type="json_parse_error",
            context={"json_string": json_string}
        )
        raise

ACTIVATION_PHRASES = ["parse json", "load json", "convert json"]