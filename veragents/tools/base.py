import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type, Union, get_type_hints
from pydantic import BaseModel, create_model, Field, ValidationError

class ToolError(Exception):
    """Base exception for tool errors."""
    def __init__(self, tool_name: str, message: str, error_type: str = "ExecutionError"):
        self.tool_name = tool_name
        self.error_type = error_type
        self.message = message
        super().__init__(f"[{tool_name}] {error_type}: {message}")

class BaseTool(ABC):
    """
    Abstract base class for all tools.
    """
    name: str
    description: str
    args_model: Type[BaseModel]

    def __init__(self, name: Optional[str] = None, description: Optional[str] = None):
        """
        Initialize the tool.
        If name/description are not provided, they should be defined in the subclass or
        extracted from the class docstring/name.
        """
        self.name = name or getattr(self, "name", self.__class__.__name__)
        self.description = description or getattr(self, "description", self.__class__.__doc__ or "")
        
        # Ensure description is clean
        if self.description:
            self.description = inspect.cleandoc(self.description)
            
        # If args_model is not defined in class, user must define it or it's empty
        if not hasattr(self, "args_model"):
            # Default to empty model if not specified (no args)
            self.args_model = create_model(f"{self.name}Args")

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Synchronous execution logic."""
        pass

    async def async_run(self, **kwargs) -> Any:
        """
        Asynchronous execution logic. 
        Default implementation calls run() in a thread if needed, 
        or subclasses can override this for native async.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.run(**kwargs))

    def validate_args(self, kwargs: Dict[str, Any], strict: bool = False) -> BaseModel:
        """
        Validate arguments against args_model.
        """
        try:
            if strict:
                return self.args_model.model_validate(kwargs, strict=True)
            else:
                return self.args_model.model_validate(kwargs)
        except ValidationError as e:
            # Format error nicely
            error_msgs = []
            for err in e.errors():
                loc = ".".join(str(l) for l in err["loc"])
                error_msgs.append(f"{loc}: {err['msg']}")
            raise ToolError(self.name, "; ".join(error_msgs), "ValidationError") from e

    def to_openai_tool(self) -> Dict[str, Any]:
        """
        Export schema to OpenAI tool format.
        """
        schema = self.args_model.model_json_schema()
        
        # OpenAI expects 'parameters' to be the schema
        # We should keep $defs if they exist for referenced types
        
        parameters = schema.copy()
        parameters.pop("title", None) # Remove top-level title
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters
            }
        }

class FunctionTool(BaseTool):
    """
    Tool derived from a python function.
    """
    def __init__(
        self, 
        func: Callable, 
        name: Optional[str] = None, 
        description: Optional[str] = None,
        args_model: Optional[Type[BaseModel]] = None
    ):
        self._func = func
        name = name or func.__name__
        description = description or func.__doc__ or ""
        
        # If args_model is not provided, generate from function signature
        if args_model is None:
            args_model = self._create_model_from_func(func)
            
        self.args_model = args_model
        super().__init__(name=name, description=description)

    def _create_model_from_func(self, func: Callable) -> Type[BaseModel]:
        """
        Generate Pydantic model from function signature and type hints.
        """
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        fields = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self" or param_name == "cls":
                continue
                
            # Determine type
            annotation = type_hints.get(param_name, Any)
            if annotation == Any and param.annotation != inspect.Parameter.empty:
                 annotation = param.annotation

            # Determine default
            if param.default == inspect.Parameter.empty:
                # Required
                fields[param_name] = (annotation, ...)
            else:
                # Optional
                fields[param_name] = (annotation, param.default)
        
        # Docstring parsing for field descriptions could be added here
        # For simplicity, we stick to basic signature parsing + docstring for tool desc
        
        return create_model(f"{func.__name__}Args", **fields)

    def run(self, **kwargs) -> Any:
        return self._func(**kwargs)
    
    async def async_run(self, **kwargs) -> Any:
        if inspect.iscoroutinefunction(self._func):
            return await self._func(**kwargs)
        return await super().async_run(**kwargs)
