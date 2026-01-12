import importlib
import inspect
from typing import Any, Dict, List, Optional, Union
from types import ModuleType

from .base import BaseTool, FunctionTool, ToolError

class ToolRegistry:
    """
    Registry for managing and dispatching tools.
    """
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, obj: Any, name: Optional[str] = None):
        """
        Register a tool or a collection of tools (Toolkit).
        
        Logic:
        1. BaseTool instance -> Register as is.
        2. BaseTool subclass -> Instantiate and register.
        3. Function -> Register as FunctionTool.
        4. Other Class/Instance -> Treat as Toolkit (scan public methods).
        """
        # 1. BaseTool Instance
        if isinstance(obj, BaseTool):
            tool_name = name or obj.name
            if tool_name in self._tools:
                print(f"Warning: Overwriting tool '{tool_name}'")
            self._tools[tool_name] = obj
            return

        # 2. BaseTool Class
        if inspect.isclass(obj) and issubclass(obj, BaseTool):
            # Instantiate
            try:
                instance = obj()
            except TypeError as e:
                # If init requires args, this might fail. 
                # For auto-registration, we assume default init.
                raise ValueError(f"Failed to instantiate tool {obj.__name__}: {e}")
            
            tool_name = name or instance.name
            self._tools[tool_name] = instance
            return

        # 3. Function
        if callable(obj) and not inspect.isclass(obj):
            ft = FunctionTool(obj, name=name)
            self._tools[ft.name] = ft
            return

        # 4. Toolkit (Class or Instance)
        # If it's a class, instantiate it
        if inspect.isclass(obj):
            try:
                instance = obj()
            except TypeError as e:
                raise ValueError(f"Failed to instantiate toolkit {obj.__name__}: {e}")
        else:
            instance = obj
            
        # Scan for public methods
        prefix = name or instance.__class__.__name__
        
        # We use inspect.getmembers to find methods.
        # predicate=inspect.ismethod ensures we get bound methods of the instance.
        for method_name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
            if not method_name.startswith("_"):
                # Register as a tool
                tool_name = f"{prefix}_{method_name}"
                
                # Check for existing tool to avoid overwrite warning spam if re-registering
                # But here we are registering fresh tools.
                
                # Create FunctionTool
                # description comes from docstring automatically in FunctionTool
                ft = FunctionTool(method, name=tool_name)
                
                # Register the sub-tool
                # We recursively call register, which will hit case #1 (BaseTool Instance)
                self.register(ft)

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def dispatch(self, tool_name: str, args: Dict[str, Any], strict: bool = False) -> Any:
        """
        Execute a tool synchronously.
        """
        tool = self.get(tool_name)
        if not tool:
            raise ToolError(tool_name, "Tool not found", "NotFoundError")
        
        # Validate args
        validated_model = tool.validate_args(args, strict=strict)
        validated_args = validated_model.model_dump()
        
        return tool.run(**validated_args)

    async def async_dispatch(self, tool_name: str, args: Dict[str, Any], strict: bool = False) -> Any:
        """
        Execute a tool asynchronously.
        """
        tool = self.get(tool_name)
        if not tool:
            raise ToolError(tool_name, "Tool not found", "NotFoundError")
            
        validated_model = tool.validate_args(args, strict=strict)
        validated_args = validated_model.model_dump()
        
        return await tool.async_run(**validated_args)

    def export_openai_tools(self) -> List[Dict[str, Any]]:
        """
        Export all registered tools as OpenAI tool definitions.
        """
        return [tool.to_openai_tool() for tool in self._tools.values()]

    def register_module(self, module: Union[str, ModuleType]):
        """
        Scan a module (and submodules) for tools.
        This imports the module. If tools are decorated with @tool, they register themselves.
        """
        if isinstance(module, str):
            importlib.import_module(module)

# Global default registry
registry = ToolRegistry()
