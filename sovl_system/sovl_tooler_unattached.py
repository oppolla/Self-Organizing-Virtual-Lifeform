from typing import Callable, Dict, Any, List, TypedDict, Optional, Union, Protocol
from dataclasses import dataclass, field, asdict
import json
import time
import os
import importlib.util
import inspect
import sqlite3
from sovl_logger import Logger
from sovl_error import ErrorManager
import re

# --- Core Data Structures ---

class ToolParameterProperties(TypedDict, total=False):
    pass # Actual definition below, after ToolParameter

class ToolParameter(TypedDict, total=False):
    type: str
    description: str
    enum: Optional[List[Union[str, int, float, bool]]]
    default: Optional[Any]
    items: Optional[Dict[str, Any]]
    properties: Optional[Dict[str, 'ToolParameter']]

ToolParameterProperties.__annotations__ = {k: ToolParameter for k in ToolParameterProperties.__annotations__}

class ToolInputParameters(TypedDict, total=False):
    type: str # Should typically be "object"
    properties: Dict[str, ToolParameter]
    required: Optional[List[str]]

class ToolExample(TypedDict):
    input: str
    output: Dict[str, Any]

class ToolSchema(TypedDict):
    name: str
    description: str
    parameters: ToolInputParameters
    examples: Optional[List[ToolExample]]
    tags: Optional[List[str]]

@dataclass
class Tool:
    name: str
    description: str
    schema: ToolSchema
    execute_func: Callable[..., Any]
    module_path: Optional[str] = None
    enabled: bool = True  # Whether the tool is enabled (from JSON schema or runtime)
    activation_phrases: List[str] = field(default_factory=list)  # Per-tool activation phrases

@dataclass
class ProcedureStep:
    action: str
    tool: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    text: Optional[str] = None
    description: Optional[str] = None

@dataclass
class ProcedureDefinition:
    name: str
    description: str
    steps: List[ProcedureStep]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    is_draft: bool = True

# --- Core Step Handler Registry (Extensible) ---
STEP_HANDLERS = {}

def register_step_handler(action):
    """Decorator to register a step handler for a given action name."""
    def decorator(func):
        STEP_HANDLERS[action] = func
        return func
    return decorator

# Register core step handlers
@register_step_handler("call_tool")
def handle_call_tool(step, context):
    tooler = context["tooler"]
    tool_name = step.get("tool")
    tool_params = step.get("params", {})
    if tool_name:
        return tooler.process_llm_tool_call(json.dumps({
            "tool_name": tool_name,
            "parameters": tool_params
        }))
    else:
        raise ValueError("Step missing 'tool' field for call_tool action.")

@register_step_handler("message")
def handle_message(step, context):
    msg = step.get("text", "")
    logger = context.get("logger")
    if logger:
        logger.info(f"Procedure message: {msg}")
    else:
        print(f"Procedure message: {msg}")
    return msg

# --- Additional Core Step Handlers ---
import time

@register_step_handler("wait")
def handle_wait(step, context):
    """Pause execution for a set time (seconds) or until a condition (not implemented)."""
    seconds = step.get("seconds")
    if seconds is not None:
        logger = context.get("logger")
        if logger:
            logger.info(f"Waiting for {seconds} seconds...")
        else:
            print(f"Waiting for {seconds} seconds...")
        time.sleep(seconds)
    # Future: implement condition-based waiting

@register_step_handler("prompt_user")
def handle_prompt_user(step, context):
    """
    Prompt the user for input and store the result in context["variables"].
    UI-agnostic: uses context["user_prompt_callback"] if available, else returns a dict for the orchestrator/UI to handle.
    """
    prompt = step.get("prompt", "Enter a value:")
    var_name = step.get("var_name", "user_input")
    if context and callable(context.get("user_prompt_callback")):
        value = context["user_prompt_callback"](prompt)
        context["variables"][var_name] = value
        return {"status": "ok", "value": value}
    else:
        # Return a special dict for the orchestrator/CLI to handle
        return {"action": "prompt_user", "prompt": prompt, "var_name": var_name}

@register_step_handler("set_variable")
def handle_set_variable(step, context):
    """Set a variable in context["variables"]."""
    variable = step.get("variable")
    value = step.get("value")
    if variable:
        context.setdefault("variables", {})[variable] = value
        logger = context.get("logger")
        if logger:
            logger.info(f"Set variable '{variable}' to {value}")
    else:
        raise ValueError("Step missing 'variable' field for set_variable action.")

@register_step_handler("use_result")
def handle_use_result(step, context):
    """Use the result of a previous step (by index) and store in a variable."""
    from_step = step.get("from_step")
    variable = step.get("variable")
    if from_step is not None and variable:
        results = context.setdefault("step_results", {})
        value = results.get(from_step)
        context.setdefault("variables", {})[variable] = value
        logger = context.get("logger")
        if logger:
            logger.info(f"Used result from step {from_step} as '{variable}': {value}")
    else:
        raise ValueError("Step missing 'from_step' or 'variable' for use_result action.")

@register_step_handler("branch")
def handle_branch(step, context):
    """Conditional logic: execute if_steps or else_steps based on a variable's value."""
    condition = step.get("condition", {})
    var = condition.get("variable")
    equals = condition.get("equals")
    variables = context.setdefault("variables", {})
    if var is not None and equals is not None:
        if variables.get(var) == equals:
            steps = step.get("if_steps", [])
        else:
            steps = step.get("else_steps", [])
        for substep in steps:
            action = substep.get('action')
            handler = STEP_HANDLERS.get(action)
            if handler:
                handler(substep, context)
    else:
        raise ValueError("branch step missing 'condition' with 'variable' and 'equals'.")

@register_step_handler("loop")
def handle_loop(step, context):
    """Repeat a set of steps a number of times."""
    times = step.get("times")
    steps = step.get("steps", [])
    if times is not None and steps:
        for i in range(times):
            logger = context.get("logger")
            if logger:
                logger.info(f"Loop iteration {i+1}/{times}")
            for substep in steps:
                action = substep.get('action')
                handler = STEP_HANDLERS.get(action)
                if handler:
                    handler(substep, context)
    else:
        raise ValueError("loop step missing 'times' or 'steps'.")

@register_step_handler("error_handler")
def handle_error_handler(step, context):
    """Try/catch error handling for a block of steps."""
    try_steps = step.get("try_steps", [])
    catch_steps = step.get("catch_steps", [])
    try:
        for substep in try_steps:
            action = substep.get('action')
            handler = STEP_HANDLERS.get(action)
            if handler:
                handler(substep, context)
    except Exception as e:
        logger = context.get("logger")
        if logger:
            logger.error(f"Error in try_steps: {e}")
        for substep in catch_steps:
            action = substep.get('action')
            handler = STEP_HANDLERS.get(action)
            if handler:
                handler(substep, context)

@register_step_handler("end")
def handle_end(step, context):
    """Explicitly end the procedure early."""
    raise StopIteration("Procedure ended by 'end' step.")

# --- Tool Registry ---

class LLMInterfaceProtocol(Protocol):
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        ...

class ToolRegistry:
    def __init__(self, logger, error_manager, tool_directory: str = "tools"):
        self.tools: Dict[str, Tool] = {}
        self.tool_directory = os.path.abspath(tool_directory)
        self.logger = logger
        self.error_manager = error_manager

    def _ensure_tool_directory_exists(self):
        if not os.path.exists(self.tool_directory):
            if self.logger:
                self.logger.info(f"Tools directory '{self.tool_directory}' not found. Creating.")
            os.makedirs(self.tool_directory, exist_ok=True)
        elif not os.path.isdir(self.tool_directory):
            if self.logger:
                self.logger.error(f"Tools directory path '{self.tool_directory}' exists but is not a directory.")
            raise NotADirectoryError(f"Tools directory path '{self.tool_directory}' exists but is not a directory.")

    def register_tool(self, tool_instance: Tool):
        if tool_instance.name in self.tools:
            if self.logger:
                self.logger.warn(f"Tool '{tool_instance.name}' from {tool_instance.module_path} (JSON: {tool_instance.schema.get('name', 'N/A')}.json) is redefining an existing tool.")
        self.tools[tool_instance.name] = tool_instance
        if self.logger:
            self.logger.info(f"Tool '{tool_instance.name}' from {tool_instance.module_path} (JSON: {tool_instance.schema.get('name', 'N/A')}.json) registered. Enabled: {tool_instance.enabled}")

    def get_tool(self, name: str) -> Optional[Tool]:
        tool = self.tools.get(name)
        if tool and tool.enabled:
            return tool
        return None

    def get_all_tool_schemas(self) -> List[ToolSchema]:
        return [tool.schema for tool in self.tools.values() if tool.enabled]

    def _validate_loaded_schema_structure(self, schema_dict: Dict[str, Any], json_file_path: str) -> bool:
        """Enhanced manual validation of the schema loaded from a JSON file."""
        def check_properties(properties, path="parameters.properties"):
            if not isinstance(properties, dict):
                if self.logger: self.logger.error(f"{path} must be a dictionary in {json_file_path}.")
                return False
            for prop_name, prop in properties.items():
                if not isinstance(prop, dict):
                    if self.logger: self.logger.error(f"Property '{prop_name}' in {path} must be a dictionary in {json_file_path}.")
                    return False
                if "type" not in prop or not isinstance(prop["type"], str):
                    if self.logger: self.logger.error(f"Property '{prop_name}' missing or invalid 'type' in {json_file_path}.")
                    return False
                if "description" not in prop or not isinstance(prop["description"], str):
                    if self.logger: self.logger.error(f"Property '{prop_name}' missing or invalid 'description' in {json_file_path}.")
                    return False
                # Recursively check nested properties for objects
                if prop["type"] == "object" and "properties" in prop:
                    if not check_properties(prop["properties"], path=f"{path}.{prop_name}.properties"):
                        return False
            return True

        if not isinstance(schema_dict, dict):
            if self.logger: self.logger.error(f"Schema in {json_file_path} is not a valid dictionary.")
            return False

        required_top_level_keys = ["name", "description", "parameters"]
        for key in required_top_level_keys:
            if key not in schema_dict:
                if self.logger: self.logger.error(f"Missing required key '{key}' in schema {json_file_path}.")
                return False

        if not isinstance(schema_dict.get("name"), str) or not schema_dict.get("name").strip():
            if self.logger: self.logger.error(f"Key 'name' in {json_file_path} must be a non-empty string.")
            return False
        if not isinstance(schema_dict.get("description"), str):
            if self.logger: self.logger.error(f"Key 'description' in {json_file_path} must be a string.")
            return False
        parameters_field = schema_dict.get("parameters")
        if not isinstance(parameters_field, dict):
            if self.logger: self.logger.error(f"Key 'parameters' in {json_file_path} must be a dictionary.")
            return False
        if parameters_field.get("type") != "object":
            if self.logger: self.logger.error(f"'parameters.type' in {json_file_path} must be 'object'.")
            return False
        if "properties" not in parameters_field or not isinstance(parameters_field.get("properties"), dict):
            if self.logger: self.logger.error(f"Key 'parameters.properties' in {json_file_path} must be a dictionary.")
            return False
        if not check_properties(parameters_field["properties"]):
            return False
        # Validate 'required' if present
        if "required" in parameters_field:
            required_list = parameters_field["required"]
            if not isinstance(required_list, list) or not all(isinstance(x, str) for x in required_list):
                if self.logger: self.logger.error(f"'parameters.required' in {json_file_path} must be a list of strings.")
                return False
            for req in required_list:
                if req not in parameters_field["properties"]:
                    if self.logger: self.logger.error(f"Required parameter '{req}' not found in properties in {json_file_path}.")
                    return False
        # Optionally check 'examples' and 'tags'
        if "examples" in schema_dict and not isinstance(schema_dict["examples"], list):
            if self.logger: self.logger.error(f"'examples' in {json_file_path} must be a list.")
            return False
        if "tags" in schema_dict and not (isinstance(schema_dict["tags"], list) and all(isinstance(t, str) for t in schema_dict["tags"])):
            if self.logger: self.logger.error(f"'tags' in {json_file_path} must be a list of strings.")
            return False
        return True

    def _validate_function_signature(self, func, schema_dict, tool_name, python_file_path):
        """Check that the function's signature matches the schema's required and optional parameters."""
        import inspect
        sig = inspect.signature(func)
        func_params = sig.parameters
        schema_params = schema_dict.get("parameters", {})
        properties = schema_params.get("properties", {})
        required = set(schema_params.get("required", []))
        all_schema_params = set(properties.keys())
        func_param_names = [p for p in func_params if func_params[p].kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)]
        func_param_set = set(func_param_names)
        # Check for missing required params
        missing = required - func_param_set
        if missing:
            if self.logger:
                self.logger.error(f"Function '{tool_name}' in {python_file_path} is missing required parameters: {missing}")
            return False
        # Warn about extra required params in function not in schema
        extra = func_param_set - all_schema_params
        if extra:
            if self.logger:
                self.logger.warn(f"Function '{tool_name}' in {python_file_path} has extra parameters not in schema: {extra}")
        # Warn if function uses *args or **kwargs
        for p in func_params.values():
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                if self.logger:
                    self.logger.warn(f"Function '{tool_name}' in {python_file_path} uses *args or **kwargs, which may hide parameter mismatches.")
        return True

    def load_tools_from_directory(self):
        self._ensure_tool_directory_exists()
        if not os.path.isdir(self.tool_directory):
            if self.logger:
                self.logger.error(f"Tool directory path '{self.tool_directory}' is not a valid directory after attempting to ensure its existence.")
            if self.error_manager:
                self.error_manager.record_error(
                    error=Exception("Invalid tool directory"),
                    error_type="tool_directory_error",
                    context={"tool_directory": self.tool_directory}
                )
            return

        for entry in os.listdir(self.tool_directory):
            subdir_path = os.path.join(self.tool_directory, entry)
            if os.path.isdir(subdir_path):
                base_name = entry
                json_file_path = os.path.join(subdir_path, base_name + ".json")
                python_file_path = os.path.join(subdir_path, base_name + ".py")

                if not os.path.exists(json_file_path) or not os.path.exists(python_file_path):
                    if self.logger:
                        self.logger.warn(f"Tool subdirectory {subdir_path} missing .json or .py file for tool '{base_name}'. Skipping tool.")
                    if self.error_manager:
                        self.error_manager.record_error(
                            error=FileNotFoundError(f"Missing .json or .py file for tool: {base_name}"),
                            error_type="tool_file_missing",
                            context={"subdir": subdir_path, "base_name": base_name}
                        )
                    continue

                schema_dict = None
                try:
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        schema_dict = json.load(f)
                except json.JSONDecodeError as e:
                    if self.logger:
                        self.logger.error(f"Error decoding JSON from {json_file_path}: {e}. Skipping tool.", exc_info=True)
                    if self.error_manager:
                        self.error_manager.record_error(
                            error=e,
                            error_type="tool_schema_json_decode_error",
                            context={"json_file": json_file_path}
                        )
                    continue
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error reading schema file {json_file_path}: {e}. Skipping tool.", exc_info=True)
                    if self.error_manager:
                        self.error_manager.record_error(
                            error=e,
                            error_type="tool_schema_file_read_error",
                            context={"json_file": json_file_path}
                        )
                    continue
                
                if not self._validate_loaded_schema_structure(schema_dict, json_file_path):
                    if self.error_manager:
                        self.error_manager.record_error(
                            error=ValueError(f"Invalid schema structure in {json_file_path}"),
                            error_type="tool_schema_validation_error",
                            context={"json_file": json_file_path}
                        )
                    continue

                tool_name_from_schema = schema_dict["name"]
                tool_description = schema_dict.get("description", "No description provided.")
                enabled = schema_dict.get("enabled", True)  # Read enabled from JSON, default True
                activation_phrases = []
                module_load_name = base_name
                try:
                    spec = importlib.util.spec_from_file_location(module_load_name, python_file_path)
                    if not spec or not spec.loader:
                        if self.logger: self.logger.error(f"Could not create module spec for {python_file_path}. Skipping tool '{tool_name_from_schema}'.")
                        if self.error_manager:
                            self.error_manager.record_error(
                                error=ImportError(f"Could not create module spec for {python_file_path}"),
                                error_type="tool_python_import_error",
                                context={"python_file": python_file_path, "tool_name": tool_name_from_schema}
                            )
                        continue
                    
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Try to get ACTIVATION_PHRASES from the Python module
                    if hasattr(module, "ACTIVATION_PHRASES"):
                        activation_phrases = getattr(module, "ACTIVATION_PHRASES")
                    elif "activation_phrases" in schema_dict:
                        activation_phrases = schema_dict["activation_phrases"]

                    if not hasattr(module, tool_name_from_schema) or not callable(getattr(module, tool_name_from_schema)):
                        if self.logger: 
                            self.logger.error(f"Python module {python_file_path} does not have a callable function named '{tool_name_from_schema}' (matching 'name' in {json_file_path}). Skipping tool.")
                        if self.error_manager:
                            self.error_manager.record_error(
                                error=AttributeError(f"Missing function '{tool_name_from_schema}' in {python_file_path}"),
                                error_type="tool_python_function_missing",
                                context={"python_file": python_file_path, "tool_name": tool_name_from_schema}
                            )
                        continue
                    
                    executable_func = getattr(module, tool_name_from_schema)
                    # Validate function signature
                    if not self._validate_function_signature(executable_func, schema_dict, tool_name_from_schema, python_file_path):
                        if self.logger:
                            self.logger.error(f"Tool '{tool_name_from_schema}' in {python_file_path} not registered due to function signature mismatch.")
                        continue
                    
                    tool_instance = Tool(
                        name=tool_name_from_schema, 
                        description=tool_description,
                        schema=schema_dict, 
                        execute_func=executable_func,
                        module_path=python_file_path, 
                        enabled=enabled,
                        activation_phrases=activation_phrases
                    )
                    self.register_tool(tool_instance)

                    # Merge custom step handlers from tool module
                    if hasattr(module, "STEP_HANDLERS"):
                        for action, handler in getattr(module, "STEP_HANDLERS").items():
                            STEP_HANDLERS[action] = handler
                    elif hasattr(module, "register_step_handlers"):
                        module.register_step_handlers(STEP_HANDLERS)

                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Failed to load Python module or function for tool '{tool_name_from_schema}' from {python_file_path}: {e}", exc_info=True)
                    if self.error_manager:
                        self.error_manager.record_error(
                            error=e,
                            error_type="tool_python_module_load_error",
                            context={"python_file": python_file_path, "tool_name": tool_name_from_schema}
                        )
                    continue

    def reload_tools(self):
        """Clear and reload the tool list from disk."""
        self.tools.clear()
        self.load_tools_from_directory()
        if self.logger:
            self.logger.info("Tool registry reloaded from disk.")

    def get_tools_by_tag(self, tag: str) -> List[Tool]:
        """Return all tools that have the given tag."""
        return [tool for tool in self.tools.values() if tool.schema.get('tags') and tag in tool.schema['tags']]

    def get_tools_by_tags(self, tags: List[str]) -> List[Tool]:
        """Return all tools that have any of the given tags."""
        return [tool for tool in self.tools.values() if tool.schema.get('tags') and any(t in tool.schema['tags'] for t in tags)]

    @staticmethod
    def rank_tools_by_tag_overlap(tools: List['Tool'], query_tags: List[str]) -> List['Tool']:
        """Rank tools by the number of overlapping tags with query_tags (descending)."""
        return sorted(
            tools,
            key=lambda tool: len(set(tool.schema.get('tags', [])) & set(query_tags)),
            reverse=True
        )

# --- Tooler (Orchestrator) ---

class Tooler:
    def __init__(self, tool_registry: ToolRegistry, logger, llm_interface: LLMInterfaceProtocol, procedure_manager=None, error_manager=None):
        self.tool_registry = tool_registry
        self.logger = logger
        self.llm_interface = llm_interface
        self.procedure_manager = procedure_manager
        self.error_manager = error_manager

    def get_available_tools_for_llm(self) -> List[ToolSchema]:
        return self.tool_registry.get_all_tool_schemas()

    async def process_llm_tool_call(self, tool_call_json_str: str) -> Dict[str, Any]:
        try:
            tool_call_data = json.loads(tool_call_json_str)
            tool_name = tool_call_data.get("tool_name")
            parameters_dict = tool_call_data.get("parameters", {})

            if not tool_name:
                if self.logger: self.logger.error("LLM tool call missing 'tool_name'.")
                if self.error_manager:
                    self.error_manager.record_error(
                        error=ValueError("LLM tool call missing 'tool_name'."),
                        error_type="llm_tool_call_missing_name",
                        context={"tool_call_json": tool_call_json_str}
                    )
                return {"error": "Missing 'tool_name' in LLM output."}

            tool_to_execute = self.tool_registry.get_tool(tool_name)
            if not tool_to_execute:
                if self.logger: self.logger.error(f"Tool '{tool_name}' not found in registry.")
                if self.error_manager:
                    self.error_manager.record_error(
                        error=KeyError(f"Tool '{tool_name}' not found in registry."),
                        error_type="llm_tool_not_found",
                        context={"tool_name": tool_name}
                    )
                return {"error": f"Tool '{tool_name}' not found."}

            if self.logger:
                self.logger.debug(f"Executing tool '{tool_name}' with params: {parameters_dict}")

            try:
                if inspect.iscoroutinefunction(tool_to_execute.execute_func):
                    result = await tool_to_execute.execute_func(**parameters_dict)
                else:
                    result = tool_to_execute.execute_func(**parameters_dict)
                if self.logger: self.logger.debug(f"Tool '{tool_name}' execution result: {result}")
                return {"tool_name": tool_name, "result": result}
            except TypeError as te:
                if self.logger:
                    self.logger.error(f"TypeError executing tool '{tool_name}': {te}. Parameters: {parameters_dict}.", exc_info=True)
                if self.error_manager:
                    self.error_manager.record_error(
                        error=te,
                        error_type="tool_type_error",
                        context={"tool_name": tool_name, "parameters": parameters_dict}
                    )
                return {"tool_name": tool_name, "error": f"Parameter mismatch for tool '{tool_name}': {te}"}
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
                if self.error_manager:
                    self.error_manager.record_error(
                        error=e,
                        error_type="tool_execution_error",
                        context={"tool_name": tool_name, "parameters": parameters_dict}
                    )
                return {"tool_name": tool_name, "error": str(e)}

        except json.JSONDecodeError as e:
            if self.logger: self.logger.error(f"Invalid JSON from LLM for tool call: {tool_call_json_str}", exc_info=True)
            if self.error_manager:
                self.error_manager.record_error(
                    error=e,
                    error_type="llm_tool_call_json_decode_error",
                    context={"tool_call_json": tool_call_json_str}
                )
            return {"error": "Invalid JSON tool call format."}
        except Exception as e:
            if self.logger: self.logger.error(f"Unexpected error processing tool call: {e}", exc_info=True)
            if self.error_manager:
                self.error_manager.record_error(
                    error=e,
                    error_type="llm_tool_call_unexpected_error",
                    context={"tool_call_json": tool_call_json_str}
                )
            return {"error": f"An unexpected error occurred: {str(e)}"}

    def should_inject_tool_guidance(self, user_input: str) -> bool:
        """
        Returns True if tool/procedure activation is detected for the given user input.
        Used to determine if the tool guidance section should be injected into the system prompt.
        """
        if detect_tool_activation(user_input):
            return True
        for tool in self.tool_registry.tools.values():
            if detect_tool_activation_for_tool(user_input, tool):
                return True
        return False

    def build_tool_guidance_section(self, max_examples_per_tool=1) -> str:
        """
        Build a detailed section for the system prompt listing all available tools and procedures,
        with their descriptions, activation phrases, parameters, tags, and usage examples.
        Returns an empty string if there are no enabled tools or procedures.
        """
        if not any(tool.enabled for tool in self.tool_registry.tools.values()) and (
            not self.procedure_manager or not self.procedure_manager.list_procedures(drafts_only=False)
        ):
            return ""
        lines = [
            "TOOL/PROCEDURE GUIDANCE:",
            "You have access to the following tools and procedures. Use them when a user request matches their purpose. For each tool, the name, description, activation phrases, parameters, tags, and usage examples are provided.",
            "When a user request matches a tool or procedure, respond with a tool call in the following format:",
            '{"tool_name": "<tool_name>", "parameters": { ... }}',
            "If no tool is appropriate, respond normally.\n"
        ]
        # Tools
        for tool in self.tool_registry.tools.values():
            if not tool.enabled:
                continue
            lines.append(f"- Tool: {tool.name}")
            lines.append(f"  Description: {tool.description}")
            if tool.activation_phrases:
                lines.append(f"  Activation Phrases: {', '.join(tool.activation_phrases)}")
            # Parameters
            params = tool.schema.get('parameters', {}).get('properties', {})
            required = set(tool.schema.get('parameters', {}).get('required', []))
            if params:
                lines.append(f"  Parameters:")
                for pname, pinfo in params.items():
                    ptype = pinfo.get('type', 'unknown')
                    pdesc = pinfo.get('description', '')
                    default = pinfo.get('default', None)
                    req = "required" if pname in required else "optional"
                    default_str = f", default={default}" if default is not None else ""
                    lines.append(f"    - {pname} ({ptype}, {req}{default_str}): {pdesc}")
            # Tags
            tags = tool.schema.get('tags', [])
            if tags:
                lines.append(f"  Tags: {', '.join(tags)}")
            # Examples
            examples = tool.schema.get("examples", [])[:max_examples_per_tool]
            for ex in examples:
                lines.append(f"  Example:")
                lines.append(f"    Input: {json.dumps(ex['input'])}")
                lines.append(f"    Output: {ex['output']}")
        # Procedures (if ProcedureManager is available and has procedures)
        if self.procedure_manager is not None:
            for proc_name in self.procedure_manager.list_procedures(drafts_only=False):
                proc = self.procedure_manager.get_procedure(proc_name)
                if proc: # Ensure procedure exists and is not a draft already filtered by list_procedures
                    lines.append(f"- Procedure: {proc.name}")
                    lines.append(f"  Description: {proc.description}")
                    steps = proc.steps
                    if steps:
                        lines.append(f"  Steps:")
                        for idx, step in enumerate(steps, 1):
                            desc = step.description or ''
                            tool = step.tool or ''
                            params = step.params or {}
                            lines.append(f"    {idx}. Action: {step.action}, Tool: {tool}, Params: {params} {('- ' + desc) if desc else ''}")
        return "\n".join(lines)

    # --- Contextual System Prompt Builder ---
    def build_contextual_system_prompt(self, user_input: str, base_prompt: str) -> str:
        """
        Build the system prompt for the LLM, injecting tool/procedure guidance only if activation is detected.
        Uses both global and per-tool activation detection.
        """
        activation_detected = False
        # Global activation detection
        if detect_tool_activation(user_input):
            activation_detected = True
        else:
            # Per-tool activation detection
            for tool in self.tool_registry.tools.values():
                if detect_tool_activation_for_tool(user_input, tool):
                    activation_detected = True
                    break
        if activation_detected:
            return base_prompt + "\n" + self.build_tool_guidance_section()
        else:
            return base_prompt

    def audit_and_publish_draft_procedures(self, llm=None):
        """
        Audit and publish (or discard/keep) all draft procedures using the LLM.
        Intended to be called after doctrine update (e.g., after striver/aspiration update).
        The orchestrator should call this at the appropriate time.
        """
        if self.procedure_manager and hasattr(self.procedure_manager, 'audit_draft_procedures_with_llm'):
            return self.procedure_manager.audit_draft_procedures_with_llm(llm or self.llm_interface)
        if self.logger:
            self.logger.warn("No procedure_manager or audit_draft_procedures_with_llm available in Tooler.")
        return None

    def review_and_prune_draft_procedures(self, llm):
        """
        Review all draft procedures using the LLM, decide which to publish, discard, or keep as draft.
        For those to be published, use the LLM to edit/format them, then canonicalize and save as published.
        """
        drafts = self.procedure_manager.list_procedures(drafts_only=True)
        if not drafts:
            if self.logger:
                self.logger.info("No draft procedures to review.")
            return

        draft_objs = [self.procedure_manager.get_procedure(name) for name in drafts]
        review_prompt = build_procedure_review_prompt(draft_objs)
        try:
            review_response = llm.generate(review_prompt)
            actions = json.loads(review_response)
        except Exception as e:
            if self.logger:
                self.logger.error(f"LLM review failed: {e}")
            return

        for item in actions:
            name, action = item["name"], item["action"]
            proc = self.procedure_manager.get_procedure(name)
            if not proc:
                continue
            if action == "publish":
                # LLM edit/canonicalize
                edit_prompt = build_procedure_edit_prompt(proc, PROCEDURE_TEMPLATE)
                try:
                    improved_json = llm.generate(edit_prompt)
                    improved_proc = json.loads(improved_json)
                    canonical_proc = canonicalize_procedure(improved_proc)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"LLM edit/canonicalization failed for {name}: {e}")
                    # Fallback: canonicalize original
                    canonical_proc = canonicalize_procedure(asdict(proc))
                # Save as published
                self.procedure_manager.add_procedure(ProcedureDefinition(**canonical_proc))
                self.procedure_manager.publish_procedure(canonical_proc["name"])
                if self.logger:
                    self.logger.info(f"Published procedure: {name}")
            elif action == "discard":
                self.procedure_manager.delete_procedure(name)
                if self.logger:
                    self.logger.info(f"Discarded procedure: {name}")
            else:
                if self.logger:
                    self.logger.info(f"Kept as draft: {name}")

# --- Procedure Manager (SQLite based) ---

DEFAULT_DATABASE_DIR = "storage" 
DEFAULT_DATABASE_FILE = os.path.join(DEFAULT_DATABASE_DIR, "procedures.db")

class ProcedureManager:
    def __init__(self, db_path: str, step_handlers: Dict[str, Any], logger, error_manager):
        self.db_path = db_path
        self.step_handlers = step_handlers
        self.logger = logger
        self.error_manager = error_manager
        self.conn = sqlite3.connect(self.db_path)
        self._ensure_table_exists()
        self._cache = {}

    def _ensure_table_exists(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS procedures (
                name TEXT PRIMARY KEY,
                description TEXT,
                steps TEXT,
                metadata TEXT,
                created_at REAL,
                updated_at REAL,
                is_draft INTEGER DEFAULT 1
            )
        """)
        self.conn.commit()

    def add_procedure(self, proc: ProcedureDefinition):
        now = time.time()
        proc.updated_at = now
        if not proc.created_at:
            proc.created_at = now
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO procedures (name, description, steps, metadata, created_at, updated_at, is_draft)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            proc.name,
            proc.description,
            json.dumps([asdict(step) for step in proc.steps]),
            json.dumps(proc.metadata),
            proc.created_at,
            proc.updated_at,
            int(proc.is_draft)
        ))
        self.conn.commit()
        self._cache[proc.name] = proc
        if self.logger:
            self.logger.info(f"Procedure '{proc.name}' added. Draft: {proc.is_draft}")

    def get_procedure(self, name: str) -> Optional[ProcedureDefinition]:
        if name in self._cache:
            return self._cache[name]
        cursor = self.conn.cursor()
        cursor.execute("SELECT name, description, steps, metadata, created_at, updated_at, is_draft FROM procedures WHERE name=?", (name,))
        row = cursor.fetchone()
        if row:
            steps = [ProcedureStep(**s) for s in json.loads(row[2])]
            proc = ProcedureDefinition(
                name=row[0],
                description=row[1],
                steps=steps,
                metadata=json.loads(row[3]),
                created_at=row[4],
                updated_at=row[5],
                is_draft=bool(row[6])
            )
            self._cache[name] = proc
            return proc
        return None

    def list_procedures(self, drafts_only=False) -> List[str]:
        cursor = self.conn.cursor()
        if drafts_only:
            cursor.execute("SELECT name FROM procedures WHERE is_draft=1")
        else:
            cursor.execute("SELECT name FROM procedures WHERE is_draft=0")
        names = [row[0] for row in cursor.fetchall()]
        return names

    def publish_procedure(self, name: str):
        cursor = self.conn.cursor()
        cursor.execute("UPDATE procedures SET is_draft=0 WHERE name=?", (name,))
        self.conn.commit()
        if self.logger:
            self.logger.info(f"Procedure '{name}' published.")
        if name in self._cache:
            self._cache[name].is_draft = False

    def delete_procedure(self, name: str):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM procedures WHERE name=?", (name,))
        self.conn.commit()
        self._cache.pop(name, None)
        if self.logger:
            self.logger.info(f"Procedure '{name}' deleted.")

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# --- Levenshtein Distance (Standard Library, as in sovl_shamer) ---
def levenshtein_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        return levenshtein_distance(b, a)
    if len(b) == 0:
        return len(a)
    previous_row = range(len(b) + 1)
    for i, c1 in enumerate(a):
        current_row = [i + 1]
        for j, c2 in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

# --- Command-Style Tool/Procedure Activation Detection (Strict + Fuzzy) ---

TOOL_ACTIVATION_PHRASES = [
    "run", "execute", "do", "start", "perform", "please", "can you", "could you", "would you", "i need you to", "initiate"
]

def detect_tool_activation(user_input: str, max_distance: int = 1) -> bool:
    """
    Returns True if the user input matches a command-style pattern for tool/procedure activation.
    Uses strict matching and Levenshtein distance fuzzy matching (max_distance=1 by default).
    """
    text = user_input.lower().strip()
    for phrase in TOOL_ACTIVATION_PHRASES:
        # Strict match at start
        if text.startswith(phrase):
            return True
        # Fuzzy match: allow small typos at the start
        segment = text[:len(phrase)+2]
        if levenshtein_distance(segment, phrase) <= max_distance:
            return True
    return False

# --- Per-Tool Activation Detection Helper ---
def detect_tool_activation_for_tool(user_input: str, tool: Tool, max_distance: int = 1) -> bool:
    """
    Returns True if the user input matches any of the tool's activation phrases (strict or fuzzy).
    Falls back to global TOOL_ACTIVATION_PHRASES if tool.activation_phrases is empty.
    """
    phrases = tool.activation_phrases or TOOL_ACTIVATION_PHRASES
    text = user_input.lower().strip()
    for phrase in phrases:
        if text.startswith(phrase):
            return True
        segment = text[:len(phrase)+2]
        if levenshtein_distance(segment, phrase) <= max_distance:
            return True
    return False

def handle_procedure_activation(user_input: str, procedure_manager: 'ProcedureManager', available_procedures: list = None, context: dict = None):
    """
    Integrates tool/procedure activation detection with ProcedureManager execution.
    - Checks for activation using detect_tool_activation.
    - Extracts procedure name by matching against available procedures.
    - Notifies user before execution (returns event for orchestrator/UI).
    - Executes the procedure and reports completion (returns event for orchestrator/UI).
    """
    if available_procedures is None:
        # Only consider published procedures for activation
        available_procedures = procedure_manager.list_procedures(drafts_only=False)
    if detect_tool_activation(user_input):
        text = user_input.lower()
        for proc_name in available_procedures:
            if proc_name.lower() in text:
                # Instead of print, return event for orchestrator/UI
                event = {
                    "action": "procedure_activation",
                    "message": f"Okay, I will run the '{proc_name}' procedure now.",
                    "procedure": proc_name
                }
                # Optionally, allow orchestrator to confirm before execution
                result = procedure_manager.execute_procedure(proc_name, context=context)
                return {
                    "event": event,
                    "result": result,
                    "completion_message": f"Procedure '{proc_name}' execution complete."
                }
        # No match found
        return {"action": "procedure_activation_failed", "message": "Sorry, I couldn't find a matching procedure to run."}
    return False

# --- Procedure Canonicalization Utility ---
def canonicalize_procedure(proc: dict) -> dict:
    """
    Ensure a procedure dict conforms to the canonical JSON schema for LLM and system compatibility.
    Fills in defaults for missing fields and validates step structure.
    Raises ValueError for malformed steps.
    """
    import time
    proc.setdefault("name", "unnamed_procedure")
    proc.setdefault("description", "")
    proc.setdefault("steps", [])
    proc.setdefault("metadata", {})
    proc.setdefault("created_at", time.time())
    proc.setdefault("updated_at", time.time())
    proc.setdefault("is_draft", True)
    # Validate steps
    for i, step in enumerate(proc["steps"]):
        if "action" not in step:
            raise ValueError(f"Step {i} is missing required 'action' field.")
    return proc

# --- Procedure Detection System ---
import re

class ProcedureDetector:
    """
    Detects whether a user input likely describes a procedure, using strong phrases, numbered list patterns,
    and a concise set of imperative verbs. Designed to minimize false positives while capturing most real procedures.
    Extendable for future LLM or advanced heuristic integration.
    """
    STRONG_PHRASES = [
        "as a first step", "begin with", "before you start", "do it like this", "do this exactly like", "execute it like this",
        "first,", "follow these steps", "here's a guide to", "here's how to", "here's how you can", "here's my way of doing it",
        "here's the way to", "here's what you need to do", "how to", "I need you to follow this", "I want you to do it this way",
        "I'll guide you through", "I'll show you how to", "I'll walk you through it", "initially,", "instructions:", "kick off with",
        "let me break it down for you", "let me explain how to", "let me show you how to", "let me tell you how it's done",
        "let me walk you through", "let's go through the steps to", "let's start with", "make it happen like this", "perform it this way",
        "start by", "step by step", "the first thing to do is", "the process is", "the steps are", "these are the steps to",
        "this is how to", "this is how you do it", "this is the procedure for", "this is the way to do it", "this is what I want you to do",
        "to accomplish this", "to achieve this", "to begin with", "to carry this out", "to complete this", "to do this, follow",
        "to ensure this works", "to execute this", "to finish this task", "to get everything in place for", "to get ready to",
        "to get started", "to get this done", "to lay the groundwork for", "to make it happen", "to make sure it's done right",
        "to make this work", "to perform this", "to prepare for this", "to pull this off", "to set this up", "to solve this",
        "to start off", "to succeed at this", "to tackle this"
    ]
    IMPERATIVE_VERBS = [
        "do", "run", "install", "configure", "set up", "fix", "create", "build", "start", "perform", "follow",
        "complete", "achieve", "implement", "use", "make", "get", "begin", "show", "explain", "write", "list",
        "plan", "organize", "prepare", "review", "check", "test", "verify"
    ]
    @staticmethod
    def detect_procedure(text: str, max_fuzzy_distance: int = 1) -> bool:
        text_lc = text.lower().strip()
        # Strong phrase match (strict or fuzzy)
        for phrase in ProcedureDetector.STRONG_PHRASES:
            if phrase in text_lc:
                return True
            # Fuzzy match at start
            segment = text_lc[:len(phrase)+2]
            if levenshtein_distance(segment, phrase) <= max_fuzzy_distance:
                return True
        # Numbered list pattern
        if re.search(r"^\s*\d+\.\s+", text_lc, re.MULTILINE):
            return True
        if re.search(r"^\s*step\s*\d+\s*:", text_lc, re.MULTILINE):
            return True
        if re.search(r"^\s*(first|second|third|next|then|finally)[,:\-]", text_lc, re.MULTILINE):
            return True
        # Imperative verb at start (strict)
        for verb in ProcedureDetector.IMPERATIVE_VERBS:
            if text_lc.startswith(verb + " "):
                return True
        # Formatting/structural cues: multiple lines with numbers
        lines = text_lc.splitlines()
        numbered_lines = sum(1 for line in lines if re.match(r"^\s*\d+\.\s+", line))
        if numbered_lines >= 2:
            return True
        return False

# --- Helper functions for LLM prompts (for procedure review/edit) ---
def build_procedure_review_prompt(draft_procs):
    procs_json = json.dumps([asdict(p) if hasattr(p, 'name') else p for p in draft_procs], indent=2)
    return (
        "You are reviewing draft procedures created during the last active cycle.\n"
        "For each, decide:\n"
        "  - 'publish' if it is clear, useful, and should be kept.\n"
        "  - 'discard' if it is redundant, unclear, or not useful.\n"
        "  - 'keep as draft' if it needs more work.\n"
        "Respond with a JSON list: "
        '[{"name": ..., "action": "publish|discard|keep as draft", "reason": "..."}]\n'
        f"Draft procedures:\n{procs_json}\n"
    )

def build_procedure_edit_prompt(proc, template):
    proc_json = json.dumps(asdict(proc) if hasattr(proc, 'name') else proc, indent=2)
    template_json = json.dumps(template, indent=2)
    return (
        "Here is a draft procedure:\n"
        f"{proc_json}\n\n"
        "Please review and, if needed, edit it to be clear, complete, and match this template:\n"
        f"{template_json}\n"
        "Return the improved procedure as a JSON object matching the template."
    )

PROCEDURE_TEMPLATE = {
    "name": "string",
    "description": "string",
    "steps": [
        {
            "action": "string",
            "tool": "string (optional)",
            "params": {},
            "text": "string (optional)",
            "description": "string (optional)"
        }
    ],
    "metadata": {},
    "created_at": 0,
    "updated_at": 0,
    "is_draft": False
}