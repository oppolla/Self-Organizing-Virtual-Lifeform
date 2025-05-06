from typing import Optional

def do_commandname(handler, arg: str) -> Optional[bool]:
    """
    Short description of what this command does.
    Usage: /commandname [args...]
    """
    # handler: the CommandHandler instance (gives access to sovl_system, logger, etc.)
    # arg: the raw argument string from the CLI
    print(f"Command executed with arg: {arg}")
    return None

# Optional metadata for registration and help
COMMAND_NAME = "commandname"
COMMAND_ALIASES = ["cmd"]
COMMAND_CATEGORY = "Uncategorized"
COMMAND_DESCRIPTION = "Short description for /help output."

def get_help() -> str:
    return do_commandname.__doc__ or COMMAND_DESCRIPTION
