from typing import Dict, List, Optional, Any
from .base import BaseTool  

class ToolRegistry:
    """
    VerAgents 现代工具注册表

    适配 Pydantic V2 架构。
    不再区分普通函数和对象，所有注册项必须是 BaseTool 实例。
    （使用 @tool 或 @toolkit 装饰器生成的对象）
    """

    def __init__(self):
        # 统一存储 BaseTool 对象
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool, override: bool = False):
        """
        注册工具 (无论是单函数还是 Toolkit)

        Args:
            tool: 由 @tool 或 @toolkit 生成的 BaseTool 实例
            override: 是否允许覆盖同名工具
        """
        if not isinstance(tool, BaseTool):
            raise TypeError(f"注册失败：对象必须是 BaseTool 实例，当前类型为 {type(tool)}")

        if tool.name in self._tools and not override:
            print(f"⚠️ 警告：工具 '{tool.name}' 已存在，跳过注册 (使用 override=True 强制覆盖)")
            return

        self._tools[tool.name] = tool
        print(f"✅ 工具 '{tool.name}' 已注册 ({tool.description})")

    def unregister(self, name: str):
        """注销工具"""
        if name in self._tools:
            del self._tools[name]
            print(f"🗑️ 工具 '{name}' 已注销。")
        else:
            print(f"⚠️ 工具 '{name}' 不存在。")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """获取工具实例"""
        return self._tools.get(name)

    def execute(self, name: str, tool_args: dict = None) -> Any:
        """
        执行工具

        Args:
            name: 工具名称 (LLM 返回的 name)
            tool_args: 工具参数字典 (LLM 返回的 arguments JSON)

        Returns:
            执行结果
        """
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found."

        try:
            # 兼容处理：如果没有参数，传空字典
            args = tool_args or {}
            # BaseTool.run 现在非常智能，可以直接处理字典
            return tool.run(**args)
        except Exception as e:
            # 在生产环境中，这里应该记录详细日志
            return f"Error executing '{name}': {str(e)}"

    @property
    def openai_tools(self) -> List[Dict[str, Any]]:
        """
        直接生成适配 OpenAI Chat Completion API 的 tools 列表

        Usage:
            client.chat.completions.create(
                ...,
                tools=registry.openai_tools
            )
        """
        return [tool.openai_schema for tool in self._tools.values()]

    def list_tools(self) -> List[str]:
        """列出所有工具名称"""
        return list(self._tools.keys())

    def clear(self):
        """清空注册表"""
        self._tools.clear()
        print("🧹 所有工具已清空。")

    def get_tools_description(self) -> str:
        """
        获取所有工具的描述文本（用于 ReAct Agent 提示词）

        Returns:
            格式化的工具描述字符串
        """
        descriptions = []
        for tool_name, tool in self._tools.items():
            # 使用完整的 validator schema（包含 $defs）
            full_schema = tool._validator.json_schema()

            # 检查是否为 toolkit（有 discriminator 或 oneOf）
            is_toolkit = "discriminator" in full_schema or "oneOf" in full_schema

            if is_toolkit:
                # Toolkit 模式
                discriminator = full_schema.get("discriminator", {})
                mapping = discriminator.get("mapping", {})
                actions = list(mapping.keys())

                # 获取第一个 action 的参数定义
                if actions:
                    first_action = actions[0]
                    def_ref = mapping.get(first_action, "")
                    # 解析 $defs/get_weatherArgs 格式
                    def_name = def_ref.split("/")[-1] if "/" in def_ref else first_action
                    def_name = def_name.replace("#/$defs/", "")

                    # 从 $defs 中获取参数
                    defs = full_schema.get("$defs", {})
                    action_def = defs.get(def_name, {})
                    action_params = action_def.get("properties", {})

                    # 提取参数（排除 action）
                    param_list = []
                    required_params = action_def.get("required", [])

                    for param_name, param_info in action_params.items():
                        if param_name == "action":
                            continue
                        is_required = "必需" if param_name in required_params else "可选"
                        param_desc = param_info.get("title", param_name)
                        param_list.append(f"`{param_desc}`({is_required})")

                    # 构建描述
                    actions_desc = ", ".join(actions)
                    if param_list:
                        params_str = ", ".join(param_list)
                        # 构建示例：使用第一个必需参数或第一个可选参数
                        example_params = []
                        for p in action_params.keys():
                            if p != "action":
                                example_params.append(f"{p}=值")
                                if len(example_params) >= 2:
                                    break

                        example_str = ", ".join(example_params)
                        desc = (f"- **{tool_name}**: {tool.description}\n"
                                f"  - 可用操作: {actions_desc}\n"
                                f"  - 参数: {params_str}\n"
                                f"  - 格式: `{tool_name}[action={first_action}, {example_str}]`")
                    else:
                        desc = (f"- **{tool_name}**: {tool.description}\n"
                                f"  - 可用操作: {actions_desc}\n"
                                f"  - 格式: `{tool_name}[action={first_action}]`")
                else:
                    desc = f"- **{tool_name}**: {tool.description}\n  - 无可用操作"
            else:
                # 单函数工具
                properties = full_schema.get("properties", {})
                if properties:
                    params_list = []
                    required = full_schema.get("required", [])

                    for param_name, param_info in properties.items():
                        is_required = "必需" if param_name in required else "可选"
                        param_desc = param_info.get("title", param_name)
                        params_list.append(f"`{param_desc}`({is_required})")

                    params_str = ", ".join(params_list)
                    first_param = list(properties.keys())[0]

                    if len(properties) == 1:
                        desc = (f"- **{tool_name}**: {tool.description}\n"
                                f"  - 参数: {params_str}\n"
                                f"  - 格式: `{tool_name}[值]` 或 `{tool_name}[{first_param}=值]`")
                    else:
                        desc = (f"- **{tool_name}**: {tool.description}\n"
                                f"  - 参数: {params_str}\n"
                                f"  - 格式: `{tool_name}[{first_param}=值, ...]`")
                else:
                    desc = f"- **{tool_name}**: {tool.description}\n  - 格式: `{tool_name}[]`"

            descriptions.append(desc)

        return "\n".join(descriptions)


# 全局单例
global_registry = ToolRegistry()