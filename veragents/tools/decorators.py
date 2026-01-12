import functools
import inspect
from typing import Optional, Callable, Type, Union, Any

from .base import BaseTool
from .registry import ToolRegistry, registry as default_registry

def tool(
    obj: Optional[Union[Callable, Type, Any]] = None,
    *,
    name: Optional[str] = None,
    registry: Optional[ToolRegistry] = None,
):
    """
    Decorator to register a tool or toolkit.
    
    Usage:
    @tool
    def my_tool(a: int): ...
    
    @tool(name="custom_name")
    def my_tool(a: int): ...
    
    @tool
    class MyTool(BaseTool): ...

    @tool
    class FileSystem:
        # Methods become tools: FileSystem_read, FileSystem_write
        def read(self, path: str): ...
        def write(self, path: str): ...
    """
    if registry is None:
        registry = default_registry

    def wrapper(target: Any) -> Any:
        # Register the tool/toolkit
        registry.register(target, name=name)
        return target

    # Handle case where decorator is called without parens: @tool
    if obj is not None:
        return wrapper(obj)
    
    # Handle case where decorator is called with parens: @tool(name="foo")
    return wrapper

# Alias
register_tool = tool
