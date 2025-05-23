import subprocess
from sovl_logger import Logger
from sovl_error import ErrorManager

def system_command(command: str) -> dict:
    logger = Logger.get_instance()
    error_manager = ErrorManager.get_instance()
    try:
        # Restrict to safe commands (example whitelist)
        allowed_commands = ["ls", "dir", "cat", "echo", "grep"]
        if not any(command.startswith(cmd) for cmd in allowed_commands):
            raise ValueError(f"Command '{command}' not in allowed list")
        
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        logger.info(f"Executed command: {command}")
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except Exception as e:
        logger.error(f"Error executing command '{command}': {e}", exc_info=True)
        error_manager.record_error(
            error=e,
            error_type="system_command_error",
            context={"command": command}
        )
        raise

ACTIVATION_PHRASES = ["run command", "execute shell", "system command"]