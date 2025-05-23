from sovl_logger import Logger
from sovl_error import ErrorManager
import math

def calculator(expression: str) -> float:
    """
    Evaluate a mathematical expression supporting advanced math functions and constants.
    Only safe functions/constants from the math module and a few built-ins (abs, round, min, max) are allowed.
    """
    logger = Logger.get_instance()
    error_manager = ErrorManager.get_instance()
    # Allowed names: math functions/constants + a few built-ins
    allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed_names.update({
        'abs': abs,
        'round': round,
        'min': min,
        'max': max
    })
    try:
        # Optionally, allow only safe characters (numbers, letters, operators, parentheses, comma, dot, underscore, spaces)
        allowed_chars = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_., +-*/()[]{}^%<>=!\'\"\t\n")
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Invalid characters in expression")
        # Replace '^' with '**' for exponentiation if user uses caret
        safe_expr = expression.replace('^', '**')
        result = eval(safe_expr, {"__builtins__": {}}, allowed_names)
        logger.info(f"Calculated expression: {expression} = {result}")
        return result
    except Exception as e:
        logger.error(f"Error calculating expression '{expression}': {e}", exc_info=True)
        error_manager.record_error(
            error=e,
            error_type="calculator_error",
            context={"expression": expression}
        )
        raise

ACTIVATION_PHRASES = ["calculate", "compute", "do math"]