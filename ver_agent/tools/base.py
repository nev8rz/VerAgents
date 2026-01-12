import inspect
from typing import Any, Callable, Dict, List, Literal, Union, Annotated, Type
from pydantic import BaseModel, Field, PrivateAttr, create_model, TypeAdapter

class BaseTool(BaseModel):
    name: str
    description: str

    # 使用 TypeAdapter
    _validator: TypeAdapter = PrivateAttr()
    _runner: Callable = PrivateAttr()

    def __init__(self, name: str, description: str, validator: TypeAdapter, runner: Callable, **data):
        super().__init__(name=name, description=description, **data)
        self._validator = validator
        self._runner = runner

    def run(self, **kwargs) -> Any:
        # 1. 校验：无论是单参数还是带 action 的多参数，TypeAdapter 一行搞定
        validated_data = self._validator.validate_python(kwargs)

        # 2. 执行：转交具体的 runner 处理
        if isinstance(validated_data, BaseModel):
            params = validated_data.model_dump()
        else:
            # 处理 RootModel 或其他情况
            params = validated_data.model_dump() if hasattr(validated_data, 'model_dump') else validated_data

        return self._runner(params)

    @property
    def openai_schema(self) -> Dict[str, Any]:
        """获取完美的 OpenAI Schema"""
        schema = self._validator.json_schema()

        def clean(d):
            if isinstance(d, dict):
                return {k: clean(v) for k, v in d.items() if k not in ('title', '$defs')}
            return d

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": clean(schema)
            }
        }


def _func_to_model(func: Callable, enforce_action: str = None) -> Type[BaseModel]:
    """将函数签名转为 Pydantic Model，支持可选的 action 注入"""
    sig = inspect.signature(func)
    fields = {}

    for name, param in sig.parameters.items():
        if name in ('self', 'cls'):
            continue
        annotation = param.annotation if param.annotation != inspect._empty else Any
        default = param.default if param.default != inspect._empty else ...
        fields[name] = (annotation, default)

    # 关键：如果是 Router 模式，强制注入 action 字段作为 Discriminator
    if enforce_action:
        fields['action'] = (Literal[enforce_action], enforce_action)

    return create_model(f"{func.__name__}Args", **fields)


def tool(func: Callable) -> BaseTool:
    """装饰器：处理单函数工具"""
    model = _func_to_model(func)

    def runner(params: dict):
        return func(**params)

    return BaseTool(
        name=func.__name__,
        description=inspect.getdoc(func) or "",
        validator=TypeAdapter(model),
        runner=runner
    )

def toolkit(cls) -> BaseTool:
    """装饰器：将类转换为 Router 模式的工具集"""
    instance = cls()  # 实例化类，保持实例引用以支持访问下划线方法
    methods = [
        getattr(instance, m) for m in dir(instance)
        if callable(getattr(instance, m)) and not m.startswith('_')
    ]

    sub_models = []
    action_names = []

    for method in methods:
        action_name = method.__name__
        action_names.append(action_name)
        sub_models.append(_func_to_model(method, enforce_action=action_name))

    UnionType = Union[tuple(sub_models)]  # type: ignore

    RouterSchema = Annotated[UnionType, Field(discriminator='action')]

    def router_runner(params: dict):
        action = params.pop('action')
        # 通过 getattr 从实例获取方法，确保 self 上下文正确
        method = getattr(instance, action)
        return method(**params)

    return BaseTool(
        name=cls.__name__,
        description=inspect.getdoc(cls) or "Toolkit",
        validator=TypeAdapter(RouterSchema),
        runner=router_runner
    )
