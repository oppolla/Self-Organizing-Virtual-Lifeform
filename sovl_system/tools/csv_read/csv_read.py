import csv
import os
from sovl_logger import Logger
from sovl_error import ErrorManager

def csv_read(file_path: str, delimiter: str = ",", encoding: str = "utf-8") -> list:
    logger = Logger.get_instance()
    error_manager = ErrorManager.get_instance()
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        with open(file_path, 'r', encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            data = [row for row in reader]
        logger.info(f"CSV file read: {file_path}, {len(data)} rows")
        return data
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}", exc_info=True)
        error_manager.record_error(
            error=e,
            error_type="csv_read_error",
            context={"file_path": file_path, "delimiter": delimiter, "encoding": encoding}
        )
        raise

ACTIVATION_PHRASES = ["read csv", "load csv", "parse csv"]