from abc import ABC, abstractmethod
from typing import List, Optional

class Command(ABC):
    """Base class for CLI commands."""
    name: str = ""  # Command name (e.g., "help", "joke")
    category: str = "Uncategorized"  # Category for COMMAND_CATEGORIES
    aliases: List[str] = []  # Optional aliases (e.g., ["h"] for /help)
    description: str = ""  # Short description for /help

    @abstractmethod
    def execute(self, args: List[str], handler: 'CommandHandler') -> Optional[bool]:
        """Execute the command. Return True to exit CLI, None otherwise."""
        pass

    @classmethod
    def get_help(cls) -> str:
        """Return help text for the command."""
        return cls.description
