import platform
import psutil
from sovl_logger import Logger
from sovl_error import ErrorManager

def system_info(info_type: str = "all") -> dict:
    logger = Logger.get_instance()
    error_manager = ErrorManager.get_instance()
    try:
        result = {}
        if info_type in ["all", "os"]:
            result["os"] = platform.system() + " " + platform.release()
        if info_type in ["all", "cpu"]:
            result["cpu_count"] = psutil.cpu_count()
            result["cpu_usage"] = psutil.cpu_percent(interval=1)
        if info_type in ["all", "memory"]:
            mem = psutil.virtual_memory()
            result["memory_total"] = mem.total / (1024 ** 3)  # GB
            result["memory_used"] = mem.used / (1024 ** 3)   # GB
            result["memory_percent"] = mem.percent
        logger.info(f"Retrieved system info: {info_type}")
        return result
    except Exception as e:
        logger.error(f"Error retrieving system info: {e}", exc_info=True)
        error_manager.record_error(
            error=e,
            error_type="system_info_error",
            context={"info_type": info_type}
        )
        raise

ACTIVATION_PHRASES = ["system info", "computer status", "get system details"]