# SOVL Tool Development Guide

This guide outlines how to create and integrate new tools into the SOVL system using the `ToolRegistry`.

## Directory Structure

Each tool should reside in its own subdirectory within the main `tools/` directory. The subdirectory should be named after the tool itself. Inside this subdirectory, you must have two files:

1.  `<tool_name>.json`: The JSON schema defining the tool's interface.
2.  `<tool_name>.py`: The Python module containing the tool's execution logic.

**Example Structure:**

```
tools/
└── example_tool/
    ├── example_tool.json
    └── example_tool.py
└── another_tool/
    ├── another_tool.json
    └── another_tool.py
```

Here, `example_tool` and `another_tool` are the names of the tools.

## JSON Schema (`<tool_name>.json`)

The JSON file defines the tool's metadata and parameters, which the system (and potentially an LLM) uses to understand and call the tool.

**Required Fields:**

*   `name` (string): The unique name of the tool. This **must** match the subdirectory name and the base name of the `.json` and `.py` files. It also **must** match the name of the callable function in your Python module.
*   `description` (string): A clear, concise description of what the tool does, its purpose, and when it should be used.
*   `parameters` (object): Defines the input parameters the tool accepts.
    *   `type` (string): Must be `"object"`.
    *   `properties` (object): A dictionary where each key is a parameter name, and the value is an object describing that parameter.
        *   `type` (string): The JSON schema type of the parameter (e.g., `"string"`, `"integer"`, `"boolean"`, `"array"`, `"object"`).
        *   `description` (string): A clear description of the parameter.
        *   `default` (any, optional): A default value for the parameter if it's not provided.
        *   `enum` (array, optional): A list of allowed values for the parameter.
        *   `items` (object, optional): If `type` is `"array"`, this describes the type of items in the array.
        *   `properties` (object, optional): If `type` is `"object"`, this describes the nested properties.
    *   `required` (array of strings, optional): A list of parameter names from `properties` that are mandatory for the tool to function.

**Optional Fields:**

*   `examples` (array of objects, optional): A list of usage examples. Each example object should have:
    *   `input` (string/object): Example input for the tool (often a JSON string or an object representing the parameters).
    *   `output` (object): The expected output or a description of the result.
*   `tags` (array of strings, optional): Keywords or categories to help in discovering or grouping tools.
*   `activation_phrases` (array of strings, optional): Specific phrases that, if detected in user input, might suggest this tool should be used. If not provided here, they can be defined in the Python module.
*   `enabled` (boolean, optional): Defaults to `true`. Set to `false` to disable the tool from being loaded by the `ToolRegistry`.

### JSON Example (`tools/example_tool/example_tool.json`):

```json
{
  "name": "example_tool",
  "description": "A simple example tool that adds two numbers.",
  "parameters": {
    "type": "object",
    "properties": {
      "number1": {
        "type": "integer",
        "description": "The first number."
      },
      "number2": {
        "type": "integer",
        "description": "The second number."
      }
    },
    "required": ["number1", "number2"]
  },
  "examples": [
    {
      "input": "{\"number1\": 5, \"number2\": 3}",
      "output": {"result": 8}
    }
  ],
  "tags": ["math", "example"],
  "activation_phrases": ["add numbers", "calculate sum"],
  "enabled": true
}
```

## Python Module (`<tool_name>.py`)

The Python module contains the actual logic that gets executed when the tool is called.

**Core Requirements:**

1.  **Callable Function:** You must define a Python function with the exact same name as specified in the `name` field of your JSON schema (and thus, the same as the file/directory base name).
    *   This function will receive the parameters defined in the JSON schema as keyword arguments.
    *   It can be a regular synchronous function (`def example_tool(...)`) or an asynchronous function (`async def example_tool(...)`). The `Tooler` will handle `await`ing it if it's async.

**Optional Elements:**

1.  **`ACTIVATION_PHRASES` (list of strings):**
    *   You can define a module-level constant `ACTIVATION_PHRASES` if you prefer to manage these phrases in Python rather than in the JSON schema. If both are present, the `ToolRegistry` prioritizes phrases from the Python module.
    *   Example: `ACTIVATION_PHRASES = ["add numbers for me", "sum these values"]`

2.  **Custom Step Handlers (Advanced):**
    *   If your tool needs to introduce new types of steps for the `ProcedureManager`, you can define them in your Python module.
    *   You can either define a dictionary `STEP_HANDLERS = {"my_custom_action": my_handler_func}` or a function `register_step_handlers(global_step_handlers_dict)` that registers your handlers. These will be merged into the main system's step handlers.

### Python Example (`tools/example_tool/example_tool.py`):

```python
# tools/example_tool/example_tool.py

# Optional: Define activation phrases here if not in JSON
# ACTIVATION_PHRASES = ["add numbers for me", "sum these values"]

def example_tool(number1: int, number2: int):
    """
    This tool adds two numbers.
    The parameters (number1, number2) are automatically passed
    from the LLM call based on the JSON schema.
    """
    # You can add logging using self.logger if this function was part of a class
    # that received a logger instance. For standalone tool functions,
    # standard print or Python's logging module can be used for debugging
    # during development, but the system's logger won't be automatically injected here.
    print(f"Executing example_tool with: number1={number1}, number2={number2}")
    
    result = number1 + number2
    
    # The return value will be sent back to the LLM or calling process.
    # It should typically be JSON-serializable.
    return {"result": result, "message": f"The sum of {number1} and {number2} is {result}."}

# Example of a custom step handler (advanced, optional)
# def my_custom_action_handler(step, context):
#     print(f"Handling my_custom_action with step: {step} and context: {context}")
#     return "Custom action executed"
#
# STEP_HANDLERS = {
#    "my_custom_action": my_custom_action_handler
# }
```

## Module Loading

The `ToolRegistry` automatically discovers and loads tools by:

1.  Scanning the `tool_directory` (e.g., `tools/`).
2.  Iterating through each subdirectory.
3.  For each subdirectory (e.g., `example_tool/`):
    *   It expects to find `<subdir_name>.json` (e.g., `example_tool.json`) and `<subdir_name>.py` (e.g., `example_tool.py`).
    *   The Python module is loaded using the subdirectory name as the module name (e.g., `example_tool`). No special Python package structure (like `__init__.py` in the `tools/` root or `sovl_tool_modules` prefix) is required for the tool modules themselves.

Ensure your file and function names strictly adhere to these conventions for successful loading.
The `ToolRegistry` also validates the JSON schema structure and the Python function signature against the schema's required parameters. 