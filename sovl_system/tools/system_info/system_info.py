import platform
import os
import sys
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
            result["cpu_count"] = os.cpu_count()
            # CPU usage is not available in stdlib; mark as not available
            result["cpu_usage"] = "Not available (requires psutil)"
        if info_type in ["all", "memory"]:
            if sys.platform == "linux" or sys.platform == "linux2":
                try:
                    with open('/proc/meminfo', 'r') as f:
                        meminfo = f.read()
                    mem_total = int([x for x in meminfo.splitlines() if 'MemTotal' in x][0].split()[1]) * 1024
                    mem_free = int([x for x in meminfo.splitlines() if 'MemAvailable' in x][0].split()[1]) * 1024
                    mem_used = mem_total - mem_free
                    result["memory_total"] = mem_total / (1024 ** 3)  # GB
                    result["memory_used"] = mem_used / (1024 ** 3)   # GB
                    result["memory_percent"] = round((mem_used / mem_total) * 100, 2)
                except Exception:
                    result["memory_total"] = result["memory_used"] = result["memory_percent"] = "Not available"
            elif sys.platform == "darwin":
                try:
                    import subprocess
                    vm_stat = subprocess.check_output(["vm_stat"]).decode("utf-8")
                    lines = vm_stat.split("\n")
                    page_size = int([l for l in lines if "page size of" in l][0].split(" ")[-2])
                    pages_free = int([l for l in lines if l.startswith("Pages free")][0].split(":")[1].strip().replace('.', ''))
                    pages_inactive = int([l for l in lines if l.startswith("Pages inactive")][0].split(":")[1].strip().replace('.', ''))
                    pages_speculative = int([l for l in lines if l.startswith("Pages speculative")][0].split(":")[1].strip().replace('.', ''))
                    pages_used = int([l for l in lines if l.startswith("Pages active")][0].split(":")[1].strip().replace('.', ''))
                    pages_wired = int([l for l in lines if l.startswith("Pages wired down")][0].split(":")[1].strip().replace('.', ''))
                    mem_free = (pages_free + pages_inactive + pages_speculative) * page_size
                    mem_used = (pages_used + pages_wired) * page_size
                    mem_total = mem_free + mem_used
                    result["memory_total"] = mem_total / (1024 ** 3)
                    result["memory_used"] = mem_used / (1024 ** 3)
                    result["memory_percent"] = round((mem_used / mem_total) * 100, 2)
                except Exception:
                    result["memory_total"] = result["memory_used"] = result["memory_percent"] = "Not available"
            elif sys.platform == "win32":
                try:
                    import ctypes
                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                        ]
                    memoryStatus = MEMORYSTATUSEX()
                    memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memoryStatus))
                    mem_total = memoryStatus.ullTotalPhys
                    mem_free = memoryStatus.ullAvailPhys
                    mem_used = mem_total - mem_free
                    result["memory_total"] = mem_total / (1024 ** 3)
                    result["memory_used"] = mem_used / (1024 ** 3)
                    result["memory_percent"] = round((mem_used / mem_total) * 100, 2)
                except Exception:
                    result["memory_total"] = result["memory_used"] = result["memory_percent"] = "Not available"
            else:
                result["memory_total"] = result["memory_used"] = result["memory_percent"] = "Not available"
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