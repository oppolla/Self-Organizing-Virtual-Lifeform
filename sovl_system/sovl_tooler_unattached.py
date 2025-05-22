from typing import Callable, Dict, Any, List, TypedDict, Optional, Union
from dataclasses import dataclass, field
import json
import time # For ProcedureDefinition timestamps
import os # For ProcedureManager db path
import importlib.util
import inspect # To inspect function signatures or for alternative ways to find callables
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

@dataclass
class ProcedureStep:
    tool_name: str
    parameters: Dict[str, Any]
    description: Optional[str] = None

@dataclass
class ProcedureDefinition:
    name: str
    description: str
    steps: List[ProcedureStep]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[float] = field(default_factory=time.time)
    updated_at: Optional[float] = field(default_factory=time.time)

# --- Tool Registry ---

class ToolRegistry:
    def __init__(self, tool_directory: str = "tools", logger=None, error_manager=None):
        self.tools: Dict[str, Tool] = {}
        self.tool_directory = os.path.abspath(tool_directory)
        self.logger = logger if logger is not None else Logger.get_instance()
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
            self.logger.info(f"Tool '{tool_instance.name}' from {tool_instance.module_path} (JSON: {tool_instance.schema.get('name', 'N/A')}.json) registered.")

    def get_tool(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def get_all_tool_schemas(self) -> List[ToolSchema]:
        return [tool.schema for tool in self.tools.values()]

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

        for filename in os.listdir(self.tool_directory):
            if filename.endswith(".json"):
                json_file_path = os.path.join(self.tool_directory, filename)
                base_name = filename[:-5] 
                python_file_name = base_name + ".py"
                python_file_path = os.path.join(self.tool_directory, python_file_name)

                if not os.path.exists(python_file_path):
                    if self.logger:
                        self.logger.warn(f"Found schema file {json_file_path}, but corresponding Python file {python_file_path} is missing. Skipping tool.")
                    if self.error_manager:
                        self.error_manager.record_error(
                            error=FileNotFoundError(f"Missing Python file for tool: {python_file_path}"),
                            error_type="tool_python_file_missing",
                            context={"json_file": json_file_path, "python_file": python_file_path}
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
                
                module_load_name = f"sovl_tool_modules.{base_name}"
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
                        module_path=python_file_path 
                    )
                    self.register_tool(tool_instance)

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
    def __init__(self, tool_registry: ToolRegistry, logger=None, llm_interface=None, procedure_manager=None, error_manager=None):
        self.tool_registry = tool_registry
        self.logger = logger if logger is not None else Logger.get_instance()
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

    def detect_and_define_procedure(self, user_query: str, conversation_history: List[Any]):
        if self.logger:
            self.logger.info("Procedure detection triggered (conceptual).")
        if not self.procedure_manager:
            msg = "ProcedureManager not available for defining procedure."
            if self.logger: self.logger.warn(msg)
            return msg
        
        return "Procedure detection logic not fully implemented yet."

# --- Procedure Manager (SQLite based) ---

DEFAULT_DATABASE_DIR = "database" 
DEFAULT_DATABASE_FILE = os.path.join(DEFAULT_DATABASE_DIR, "procedures.db")

class ProcedureManager:
    def __init__(self, db_path: str = DEFAULT_DATABASE_FILE, logger=None, error_manager=None):
        self.db_path = os.path.abspath(db_path)
        self.logger = logger if logger is not None else Logger.get_instance()
        self.error_manager = error_manager
        self._ensure_db_path_exists()
        self._create_table_if_not_exists()

    def _ensure_db_path_exists(self):
        db_dir = os.path.dirname(self.db_path)
        if not os.path.exists(db_dir):
            if self.logger: self.logger.info(f"Database directory '{db_dir}' not found. Creating.")
            try:
                os.makedirs(db_dir, exist_ok=True)
            except Exception as e:
                if self.logger: self.logger.error(f"Failed to create database directory '{db_dir}': {e}", exc_info=True)
                if self.error_manager:
                    self.error_manager.record_error(
                        error=e,
                        error_type="procedure_db_dir_create_error",
                        context={"db_dir": db_dir}
                    )
                raise
        elif not os.path.isdir(db_dir):
             if self.logger: self.logger.error(f"Database path '{db_dir}' exists but is not a directory.")
             if self.error_manager:
                self.error_manager.record_error(
                    error=NotADirectoryError(f"Database path '{db_dir}' exists but is not a directory."),
                    error_type="procedure_db_dir_not_directory",
                    context={"db_dir": db_dir}
                )
             raise NotADirectoryError(f"Database path '{db_dir}' exists but is not a directory.")

    def _create_table_if_not_exists(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS procedures (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    steps TEXT, 
                    metadata TEXT, 
                    created_at REAL,
                    updated_at REAL
                )
            """)
            conn.commit()
            if self.logger: self.logger.debug(f"Procedures table ensured at {self.db_path}.")
        except sqlite3.Error as e:
            if self.logger: self.logger.error(f"SQLite error creating procedures table at {self.db_path}: {e}", exc_info=True)
            if self.error_manager:
                self.error_manager.record_error(
                    error=e,
                    error_type="procedure_db_table_create_error",
                    context={"db_path": self.db_path}
                )
        finally:
            if conn: conn.close()

    def save_procedure(self, proc_def: ProcedureDefinition) -> bool:
        conn = None
        now = time.time()
        proc_def.updated_at = now
        if proc_def.created_at is None: 
            proc_def.created_at = now

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            steps_json = json.dumps([step.__dict__ for step in proc_def.steps])
            metadata_json = json.dumps(proc_def.metadata)

            cursor.execute("""
                INSERT INTO procedures (name, description, steps, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description=excluded.description,
                    steps=excluded.steps,
                    metadata=excluded.metadata,
                    updated_at=excluded.updated_at
            """, (
                proc_def.name,
                proc_def.description,
                steps_json,
                metadata_json,
                proc_def.created_at,
                proc_def.updated_at
            ))
            conn.commit()
            if self.logger: self.logger.info(f"Procedure '{proc_def.name}' saved successfully to {self.db_path}.")
            return True
        except sqlite3.Error as e:
            if self.logger: self.logger.error(f"SQLite error saving procedure '{proc_def.name}' to {self.db_path}: {e}", exc_info=True)
            if self.error_manager:
                self.error_manager.record_error(
                    error=e,
                    error_type="procedure_db_save_error",
                    context={"db_path": self.db_path, "procedure_name": proc_def.name}
                )
            return False
        finally:
            if conn: conn.close()

    def get_procedure(self, name: str) -> Optional[ProcedureDefinition]:
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name, description, steps, metadata, created_at, updated_at FROM procedures WHERE name=?", (name,))
            row = cursor.fetchone()
            if row:
                steps_list_of_dicts = json.loads(row[2])
                steps = [ProcedureStep(**s_dict) for s_dict in steps_list_of_dicts]
                metadata_dict = json.loads(row[3])
                return ProcedureDefinition(
                    name=row[0],
                    description=row[1],
                    steps=steps,
                    metadata=metadata_dict,
                    created_at=row[4],
                    updated_at=row[5]
                )
            if self.logger: self.logger.debug(f"Procedure '{name}' not found in {self.db_path}.")
            return None
        except sqlite3.Error as e:
            if self.logger: self.logger.error(f"SQLite error fetching procedure '{name}' from {self.db_path}: {e}", exc_info=True)
            if self.error_manager:
                self.error_manager.record_error(
                    error=e,
                    error_type="procedure_db_fetch_error",
                    context={"db_path": self.db_path, "procedure_name": name}
                )
            return None
        except json.JSONDecodeError as e: 
            if self.logger: self.logger.error(f"JSON decode error for procedure '{name}' from DB {self.db_path}: {e}", exc_info=True)
            if self.error_manager:
                self.error_manager.record_error(
                    error=e,
                    error_type="procedure_db_json_decode_error",
                    context={"db_path": self.db_path, "procedure_name": name}
                )
            return None
        finally:
            if conn: conn.close()

    def list_procedures(self) -> List[Dict[str, Any]]: 
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name, description, updated_at FROM procedures ORDER BY name")
            procedures = [
                {
                    "name": row[0],
                    "description": row[1],
                    "updated_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(row[2])) if row[2] else "N/A"
                }
                for row in cursor.fetchall()
            ]
            if self.logger: self.logger.debug(f"Listed {len(procedures)} procedures from {self.db_path}.")
            return procedures
        except sqlite3.Error as e:
            if self.logger: self.logger.error(f"SQLite error listing procedures from {self.db_path}: {e}", exc_info=True)
            if self.error_manager:
                self.error_manager.record_error(
                    error=e,
                    error_type="procedure_db_list_error",
                    context={"db_path": self.db_path}
                )
            return []
        finally:
            if conn: conn.close()

    def delete_procedure(self, name: str) -> bool:
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM procedures WHERE name=?", (name,))
            conn.commit()
            deleted_count = cursor.rowcount
            if deleted_count > 0:
                if self.logger: self.logger.info(f"Procedure '{name}' deleted successfully from {self.db_path}.")
            elif deleted_count == 0: 
                if self.logger: self.logger.warn(f"Attempted to delete procedure '{name}' from {self.db_path}, but it was not found.")
            return deleted_count > 0
        except sqlite3.Error as e:
            if self.logger: self.logger.error(f"SQLite error deleting procedure '{name}' from {self.db_path}: {e}", exc_info=True)
            if self.error_manager:
                self.error_manager.record_error(
                    error=e,
                    error_type="procedure_db_delete_error",
                    context={"db_path": self.db_path, "procedure_name": name}
                )
            return False
        finally:
            if conn: conn.close()

# --- Procedural Memory System (Third Pillar) ---

class ProceduralMemory:
    def __init__(self, db_path: str, tooler, logger, error_manager):
        self.db_path = db_path
        self.tooler = tooler
        self.logger = logger
        self.error_manager = error_manager
        self._ensure_table_exists()
        self._cache = {}

    def _ensure_table_exists(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS procedures (
                name TEXT PRIMARY KEY,
                description TEXT,
                steps TEXT,
                metadata TEXT,
                created_at REAL,
                updated_at REAL
            )
        """)
        conn.commit()
        conn.close()

    def add_procedure(self, name: str, description: str, steps: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None):
        now = time.time()
        record = {
            "name": name,
            "description": description,
            "steps": steps,
            "metadata": metadata or {},
            "created_at": now,
            "updated_at": now
        }
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO procedures (name, description, steps, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            name, description, json.dumps(steps), json.dumps(metadata or {}), now, now
        ))
        conn.commit()
        conn.close()
        self._cache[name] = record
        if self.logger:
            self.logger.info(f"Procedure '{name}' added to procedural memory.")

    def get_procedure(self, name: str) -> Optional[Dict[str, Any]]:
        if name in self._cache:
            return self._cache[name]
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name, description, steps, metadata, created_at, updated_at FROM procedures WHERE name=?", (name,))
        row = cursor.fetchone()
        conn.close()
        if row:
            record = {
                "name": row[0],
                "description": row[1],
                "steps": json.loads(row[2]),
                "metadata": json.loads(row[3]),
                "created_at": row[4],
                "updated_at": row[5]
            }
            self._cache[name] = record
            return record
        return None

    def list_procedures(self) -> List[str]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM procedures")
        names = [row[0] for row in cursor.fetchall()]
        conn.close()
        return names

    def delete_procedure(self, name: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM procedures WHERE name=?", (name,))
        conn.commit()
        conn.close()
        self._cache.pop(name, None)
        if self.logger:
            self.logger.info(f"Procedure '{name}' deleted from procedural memory.")
        return cursor.rowcount > 0

    def update_procedure(self, name: str, **fields):
        proc = self.get_procedure(name)
        if not proc:
            raise ValueError(f"Procedure '{name}' not found.")
        proc.update(fields)
        proc['updated_at'] = time.time()
        self.add_procedure(proc['name'], proc['description'], proc['steps'], proc['metadata'])

    def execute_procedure(self, name: str, context: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None):
        proc = self.get_procedure(name)
        if not proc:
            if self.logger:
                self.logger.error(f"Procedure '{name}' not found.")
            return
        for step in proc['steps']:
            if step['action'] == 'call_tool':
                tool_name = step['tool']
                tool_params = step.get('params', {})
                # Optionally merge in params/context
                if self.logger:
                    self.logger.info(f"Executing tool '{tool_name}' with params {tool_params}")
                try:
                    result = self.tooler.process_llm_tool_call(json.dumps({
                        "tool_name": tool_name,
                        "parameters": tool_params
                    }))
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error executing tool '{tool_name}' in procedure '{name}': {e}")
                    if self.error_manager:
                        self.error_manager.record_error(
                            error=e,
                            error_type="procedure_tool_execution_error",
                            context={"procedure": name, "tool": tool_name, "params": tool_params}
                        )
            elif step['action'] == 'message':
                if self.logger:
                    self.logger.info(f"Procedure message: {step['text']}")
            # Add more action types as needed
        if self.logger:
            self.logger.info(f"Procedure '{name}' execution complete.")

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

def handle_procedure_activation(user_input: str, procedural_memory: 'ProceduralMemory', available_procedures: list = None):
    """
    Integrates tool/procedure activation detection with ProceduralMemory execution.
    - Checks for activation using detect_tool_activation.
    - Extracts procedure name by matching against available procedures.
    - Notifies user before execution.
    - Executes the procedure and reports completion.
    """
    if available_procedures is None:
        available_procedures = procedural_memory.list_procedures()
    if detect_tool_activation(user_input):
        text = user_input.lower()
        for proc_name in available_procedures:
            if proc_name.lower() in text:
                print(f"Okay, I will run the '{proc_name}' procedure now.")
                procedural_memory.execute_procedure(proc_name)
                print(f"Procedure '{proc_name}' execution complete.")
                return True
        print("Sorry, I couldn't find a matching procedure to run.")
        return False
    return False