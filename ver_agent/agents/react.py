"""ReAct Agent 实现 - 推理与行动结合的智能体"""

import re
from typing import Optional, Tuple, Iterator
from ..core.agent import Agent
from ..core.llm import VerAgentLLM
from ..core.config import Config
from ..core.message import Message
from ..tools.registry import ToolRegistry

# 默认 ReAct 提示词模板
DEFAULT_REACT_PROMPT = """你是一个具备推理和行动能力的 AI 助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤：

Thought: 分析问题，确定需要什么信息，制定研究策略。
Action: 选择合适的工具获取信息：
- 调用工具：`工具名[参数]` 或 `工具名[参数名=值]`
- 完成任务：`Finish[结论]`

## ⚠️ 重要提醒
1. 每次回应必须包含 Thought 和 Action 两部分
2. 工具调用格式严格遵循：工具名[参数]
3. **对于工具集（Toolkit），必须指定 action 参数！**
   - 例如：`WeatherFetcher[action=get_weather, location=北京]`
   - action 参数必须使用工具说明中列出的可用操作名称
4. 对于多参数工具，使用 `参数名=值` 格式
5. **关键：每次必须做不同的事情！查看下面的"已执行的操作"，避免重复！**
6. 只有当你确信有足够信息回答问题时，才使用 Finish
7. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 当前任务
**Question:** {question}

## 已执行的操作（重要！不要重复这些操作！）
{history}

现在开始你的推理和行动："""


class ReActAgent(Agent):
    """
    ReAct (Reasoning and Acting) Agent

    结合推理和行动的智能体，能够：
    1. 分析问题并制定行动计划
    2. 调用外部工具获取信息
    3. 基于观察结果进行推理
    4. 迭代执行直到得出最终答案

    这是一个经典的 Agent 范式，特别适合需要外部信息的任务。
    """

    def __init__(
        self,
        name: str,
        llm: VerAgentLLM,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None,
        verbose: bool = True
    ):
        """
        初始化 ReActAgent

        Args:
            name: Agent 名称
            llm: LLM 实例
            tool_registry: 工具注册表（可选，如果不提供则创建空的）
            system_prompt: 系统提示词
            config: 配置对象
            max_steps: 最大执行步数
            custom_prompt: 自定义提示词模板
            verbose: 是否显示详细过程
        """
        super().__init__(name, llm, system_prompt, config)

        # 如果没有提供 tool_registry，创建一个空的
        if tool_registry is None:
            self.tool_registry = ToolRegistry()
        else:
            self.tool_registry = tool_registry

        self.max_steps = max_steps
        self.verbose = verbose
        self.current_history: list[str] = []

        # 设置提示词模板：用户自定义优先，否则使用默认模板
        self.prompt_template = custom_prompt if custom_prompt else DEFAULT_REACT_PROMPT

    def add_tool(self, tool):
        """
        添加工具到工具注册表

        Args:
            tool: BaseTool 实例（使用 @tool 或 @toolkit 装饰器生成）
        """
        self.tool_registry.register(tool)

    def run(self, input_text: str, **kwargs) -> str:
        """
        运行 ReAct Agent（非流式，返回最终答案）

        Args:
            input_text: 用户问题
            **kwargs: 其他参数

        Returns:
            最终答案
        """
        # 遍历流式输出，显示并收集最终答案
        final_answer = ""
        for chunk in self.run_stream(input_text, **kwargs):
            if chunk:
                # 显示输出（与 run_stream 行为一致）
                print(chunk, end="", flush=True)
                final_answer = chunk
        print()  # 换行
        return final_answer

    def run_stream(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        运行 ReAct Agent（流式输出）

        Args:
            input_text: 用户问题
            **kwargs: 其他参数

        Yields:
            思考过程、观察结果、最终答案等
        """
        self.current_history = []
        current_step = 0

        if self.verbose:
            yield f"\n🤖 {self.name} 开始处理问题: {input_text}"

        while current_step < self.max_steps:
            current_step += 1
            if self.verbose:
                yield f"\n--- 第 {current_step} 步 ---"

            # 构建提示词
            tools_desc = self.tool_registry.get_tools_description()
            history_str = "\n".join(self.current_history) if self.current_history else "（暂无历史）"
            prompt = self.prompt_template.format(
                tools=tools_desc,
                question=input_text,
                history=history_str
            )

            # 调用 LLM（流式）
            messages = [{"role": "user", "content": prompt}]
            response_text = ""

            # 静默收集流式输出，稍后格式化输出
            for chunk in self.llm.think(messages, **kwargs):
                if chunk:
                    response_text += chunk

            if not response_text:
                if self.verbose:
                    yield "\n❌ 错误：LLM 未能返回有效响应。"
                break

            # 解析输出
            thought, action = self._parse_output(response_text)

            if thought and self.verbose:
                yield f"\n🤔 思考: {thought}\n"

            if not action:
                if self.verbose:
                    yield "\n⚠️ 警告：未能解析出有效的 Action，流程终止。\n"
                break

            # 检查是否完成
            if action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                if self.verbose:
                    yield f"\n🎉 最终答案: {final_answer}\n"

                # 保存到历史记录
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(final_answer, "assistant"))

                yield final_answer
                return

            # 执行工具调用
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or tool_input is None:
                self.current_history.append("Observation: 无效的 Action 格式，请检查。")
                if self.verbose:
                    yield "\n⚠️ 无效的 Action 格式\n"
                continue

            if self.verbose:
                yield f"\n🎬 行动: {tool_name}[{tool_input}]"

            # 调用工具 - 智能构建参数
            tool_args = self._build_tool_args(tool_name, tool_input)
            observation = self.tool_registry.execute(tool_name, tool_args)
            if self.verbose:
                yield f"\n👀 观察: {observation}\n"

            # 更新历史
            self.current_history.append(f"Action: {action}")
            self.current_history.append(f"Observation: {observation}")

        if self.verbose:
            yield "\n⏰ 已达到最大步数，流程终止。\n"

        final_answer = "抱歉，我无法在限定步数内完成这个任务。"

        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))

        yield final_answer

    def _parse_output(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        解析 LLM 输出，提取思考和行动

        Args:
            text: LLM 返回的文本

        Returns:
            (thought, action) 元组
        """
        # 提取思考
        thought_match = re.search(r"Thought:\s*(.*?)(?:\n|$)", text)
        thought = thought_match.group(1).strip() if thought_match else None

        # 提取行动 - 支持两种格式：
        # 1. Action: tool_name[...]
        # 2. 直接的 tool_name[...] 或 Finish[...]
        action_match = re.search(r"Action:\s*(.*?)(?:\n|$)", text)

        if not action_match:
            # 尝试匹配直接的 tool_name[...] 或 Finish[...] 格式
            # 匹配行首或换行后的内容
            direct_match = re.search(r"(?:^|\n)\s*(\w+\[.*?\])(?:\s|$)", text, re.MULTILINE)
            if direct_match:
                action = direct_match.group(1).strip()
            else:
                # 最后尝试：匹配任何 [...] 格式
                bracket_match = re.search(r"(\w+\[.*?\])", text)
                action = bracket_match.group(1).strip() if bracket_match else None
        else:
            action = action_match.group(1).strip()

        return thought, action

    def _parse_action(self, action_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        解析行动文本，提取工具名称和输入

        Args:
            action_text: Action 文本，格式如 "tool_name[input]"

        Returns:
            (tool_name, tool_input) 元组
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def _parse_action_input(self, action_text: str) -> str:
        """
        解析行动输入（用于 Finish）

        Args:
            action_text: Finish 文本，格式如 "Finish[answer]"

        Returns:
            提取的答案
        """
        match = re.match(r"\w+\[(.*)\]", action_text)
        return match.group(1) if match else ""

    def _build_tool_args(self, tool_name: str, tool_input: str) -> dict:
        """
        智能构建工具参数
        支持三种格式：
        1. JSON 格式：tool_name[{"key": "value"}] -> 解析 JSON
        2. 多参数格式：tool_name[key1=val1, key2=val2] -> 显式指定参数
        3. 简单格式：tool_name[值] -> 自动映射到第一个参数

        Args:
            tool_name: 工具名称
            tool_input: 工具输入字符串

        Returns:
            工具参数字典
        """
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return {"query": tool_input}

        # 获取完整的 validator schema（包含 $defs）
        full_schema = tool._validator.json_schema()

        # 检查是否是 toolkit 模式（有 discriminator）
        is_toolkit = "discriminator" in full_schema

        # 处理 JSON 格式
        import json
        tool_input_stripped = tool_input.strip()
        if tool_input_stripped.startswith("{") and tool_input_stripped.endswith("}"):
            try:
                # 尝试解析 JSON 格式
                json_args = json.loads(tool_input_stripped)
                if isinstance(json_args, dict):
                    return json_args
            except json.JSONDecodeError:
                pass  # 不是有效的 JSON，继续处理其他格式

        # 解析参数
        if "=" in tool_input:
            # 解析 key=value 格式
            result = {}
            parts = tool_input.split(",")
            for part in parts:
                part = part.strip()
                if "=" in part:
                    key, value = part.split("=", 1)
                    result[key.strip()] = value.strip()
            return result
        else:
            # 简单格式：toolkit 需要 action，普通工具用第一个参数
            if is_toolkit:
                # Toolkit：需要从 $defs 获取第一个 action 的参数
                discriminator = full_schema.get("discriminator", {})
                mapping = discriminator.get("mapping", {})
                actions = list(mapping.keys())
                default_action = actions[0] if actions else None

                defs = full_schema.get("$defs", {})
                if default_action and defs:
                    # 获取 def 名称（可能需要从 mapping 中解析）
                    def_ref = mapping.get(default_action, "")
                    # 解析 "#/$defs/get_weatherArgs" -> "get_weatherArgs"
                    def_name = def_ref.split("/")[-1] if "/" in def_ref else default_action
                    # 尝试多种方式查找定义
                    if def_name in defs:
                        action_def = defs[def_name]
                    elif default_action in defs:
                        action_def = defs[default_action]
                    else:
                        # 尝试在所有 defs 中查找
                        for key in defs:
                            if key.endswith(default_action) or default_action in key:
                                action_def = defs[key]
                                break
                        else:
                            action_def = {}

                    props = action_def.get("properties", {})
                    param_names = [k for k in props.keys() if k != "action"]

                    if param_names:
                        return {"action": default_action, param_names[0]: tool_input.strip()}
                    else:
                        return {"action": default_action, "input": tool_input.strip()}
                return {"action": "action", "input": tool_input.strip()}
            else:
                # 普通工具：获取第一个参数名
                properties = full_schema.get("properties", {})
                if properties:
                    first_param = list(properties.keys())[0]
                    return {first_param: tool_input.strip()}
                else:
                    return {"query": tool_input.strip()}
