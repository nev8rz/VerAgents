"""Reflection Agent 实现 - 自我反思与迭代优化的智能体"""

import re
from typing import Optional, Iterator, Dict
from ..core.agent import Agent
from ..core.llm import VerAgentLLM
from ..core.config import Config
from ..core.message import Message
from ..tools.registry import ToolRegistry


# 默认 Reflection 提示词模板
DEFAULT_INITIAL_PROMPT = """你是一个专业的任务执行助手。请根据以下要求完成任务。

## 可用工具
{tools}

## 当前任务
{task}

请提供一个完整、准确的回答。

## 回答格式
- 如果需要调用工具：Thought（思考）→ Action（工具调用）
- 如果可以直接回答：直接给出答案
- 完成后使用：Finish[你的答案]

现在开始执行任务："""

DEFAULT_REFLECT_PROMPT = """你是一个严谨的质量评审员。请仔细审查以下回答，并找出可能的问题或改进空间。

## 原始任务
{task}

## 当前回答
{content}

## 评审标准
1. 回答是否完整、准确地解决了任务？
2. 是否有遗漏的关键信息？
3. 逻辑是否清晰、合理？
4. 语言表达是否准确、专业？

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。

## 输出格式
如果回答已经很好，请回答："无需改进"。
否则，请按以下格式输出：
- 问题：[问题描述]
- 建议：[改进建议]"""

DEFAULT_REFINE_PROMPT = """你是一个专业的任务执行助手。请根据反馈意见改进你的回答。

## 可用工具
{tools}

## 原始任务
{task}

## 上一轮回答
{last_attempt}

## 反馈意见
{feedback}

请提供一个改进后的回答。

## 回答格式
- 如果需要调用工具：Thought（思考）→ Action（工具调用）
- 如果可以直接回答：直接给出答案
- 完成后使用：Finish[你的答案]

现在开始改进回答："""


class SimpleMemory:
    """
    简单的短期记忆模块，用于存储 Reflection Agent 的执行与反思轨迹。
    """

    def __init__(self):
        self.records: list[dict] = []

    def add_record(self, record_type: str, content: str):
        """向记忆中添加一条新记录

        Args:
            record_type: 记录类型（"execution" 或 "reflection"）
            content: 记录内容
        """
        self.records.append({"type": record_type, "content": content})

    def get_trajectory(self) -> str:
        """将所有记忆记录格式化为一个连贯的字符串文本"""
        if not self.records:
            return "（暂无历史记录）"

        parts = []
        for i, record in enumerate(self.records):
            if record["type"] == "execution":
                parts.append(f"--- 第 {len([r for r in self.records[:i+1] if r['type'] == 'execution'])} 轮回答 ---\n{record['content']}")
            elif record["type"] == "reflection":
                parts.append(f"--- 评审意见 ---\n{record['content']}")

        return "\n\n".join(parts)

    def get_last_execution(self) -> str:
        """获取最近一次的执行结果"""
        for record in reversed(self.records):
            if record["type"] == "execution":
                return record["content"]
        return ""

    def get_last_reflection(self) -> str:
        """获取最近一次的反思结果"""
        for record in reversed(self.records):
            if record["type"] == "reflection":
                return record["content"]
        return ""

    def clear(self):
        """清空记忆"""
        self.records.clear()


class ReflectionAgent(Agent):
    """
    Reflection Agent - 自我反思与迭代优化的智能体

    基于 Reflexion 论文的实现，通过自我反思和迭代改进来提升输出质量。

    核心思想：
    1. 执行初始任务
    2. 对结果进行自我反思
    3. 根据反思结果进行优化
    4. 迭代改进直到满意或达到最大迭代次数

    工作流程：
        Initial → Reflect → Refine → Reflect → Refine → ... → Final Answer

    特别适合：
    - 代码生成与优化
    - 文档写作
    - 分析报告
    - 需要多轮迭代完善的任务
    """

    def __init__(
        self,
        name: str,
        llm: VerAgentLLM,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        custom_prompts: Optional[Dict[str, str]] = None,
        verbose: bool = True
    ):
        """
        初始化 ReflectionAgent

        Args:
            name: Agent 名称
            llm: LLM 实例
            tool_registry: 工具注册表（可选）
            system_prompt: 系统提示词
            config: 配置对象
            max_iterations: 最大迭代次数
            custom_prompts: 自定义提示词模板 {"initial": "", "reflect": "", "refine": ""}
            verbose: 是否显示详细过程
        """
        super().__init__(name, llm, system_prompt, config)

        # 如果没有提供 tool_registry，创建一个空的
        if tool_registry is None:
            self.tool_registry = ToolRegistry()
        else:
            self.tool_registry = tool_registry

        self.max_iterations = max_iterations
        self.verbose = verbose
        self.memory = SimpleMemory()

        # 设置提示词模板：用户自定义优先，否则使用默认模板
        prompts = custom_prompts if custom_prompts else {}
        self.initial_prompt = prompts.get("initial", DEFAULT_INITIAL_PROMPT)
        self.reflect_prompt = prompts.get("reflect", DEFAULT_REFLECT_PROMPT)
        self.refine_prompt = prompts.get("refine", DEFAULT_REFINE_PROMPT)

    def add_tool(self, tool):
        """添加工具到工具注册表"""
        self.tool_registry.register(tool)

    def run(self, input_text: str, **kwargs) -> str:
        """
        运行 Reflection Agent（非流式，返回最终答案）

        Args:
            input_text: 任务描述
            **kwargs: 其他参数

        Returns:
            最终优化后的结果
        """
        final_answer = ""
        for chunk in self.run_stream(input_text, **kwargs):
            # 收集最终答案（最后一个不包含特殊字符的 chunk）
            if not chunk.startswith("\n") and not chunk.startswith("=") and not chunk.startswith("---"):
                final_answer = chunk
        return final_answer

    def run_stream(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        运行 Reflection Agent（流式输出）

        Args:
            input_text: 任务描述
            **kwargs: 其他参数

        Yields:
            执行过程的输出
        """
        # 重置记忆
        self.memory.clear()

        if self.verbose:
            yield f"\n🤖 {self.name} 开始处理任务: {input_text}"

        # ============ 阶段 1: 初始执行 ============
        if self.verbose:
            yield "\n" + "="*50 + "\n"
            yield "📝 阶段 1: 初始尝试\n"
            yield "="*50 + "\n"

        initial_result = yield from self._execute_phase(input_text, self.initial_prompt, **kwargs)
        self.memory.add_record("execution", initial_result)

        if self.verbose:
            yield f"\n✅ 初始回答完成"

        # ============ 阶段 2: 迭代优化 ============
        for i in range(self.max_iterations):
            if self.verbose:
                yield "\n" + "="*50 + "\n"
                yield f"🔍 阶段 2.{i + 1}: 反思与优化\n"
                yield "="*50 + "\n"

            # a. 反思
            if self.verbose:
                yield "\n→ 正在进行反思...\n"

            reflection = yield from self._reflection_phase(input_text, **kwargs)
            self.memory.add_record("reflection", reflection)

            if self.verbose:
                yield f"\n📊 反思结果:\n{reflection}\n"

            # b. 检查是否需要停止
            if self._is_satisfactory(reflection):
                if self.verbose:
                    yield "\n✅ 反思认为结果已满意，任务完成。\n"
                break

            # c. 优化
            if self.verbose:
                yield "\n→ 正在进行优化...\n"

            refined_result = yield from self._execute_phase(input_text, self.refine_prompt, **kwargs)
            self.memory.add_record("execution", refined_result)

            if self.verbose:
                yield f"\n🔄 优化完成\n"

        # ============ 阶段 3: 返回最终结果 ============
        final_result = self.memory.get_last_execution()

        if self.verbose:
            yield "\n" + "="*50 + "\n"
            yield "🎉 最终结果\n"
            yield "="*50 + "\n"
            yield final_result

        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_result, "assistant"))

        yield final_result

    def _execute_phase(self, task: str, prompt_template: str, **kwargs) -> Iterator[str]:
        """
        执行阶段（初始或优化）

        Args:
            task: 任务描述
            prompt_template: 提示词模板
            **kwargs: LLM 调用参数

        Yields:
            执行过程的输出

        Returns:
            执行结果
        """
        tools_desc = self.tool_registry.get_tools_description()
        last_result = self.memory.get_last_execution()
        trajectory = self.memory.get_trajectory()

        # 构建提示词
        if "上一轮回答" in prompt_template or "last_attempt" in prompt_template:
            # 优化阶段的提示词
            last_reflection = self.memory.get_last_reflection()
            prompt = prompt_template.format(
                task=task,
                tools=tools_desc,
                last_attempt=last_result,
                feedback=last_reflection
            )
        else:
            # 初始阶段的提示词
            prompt = prompt_template.format(
                task=task,
                tools=tools_desc
            )

        # 添加历史轨迹（如果有）
        if trajectory and "（暂无历史记录）" not in trajectory:
            prompt += f"\n\n## 历史记录\n{trajectory}"

        # 调用 LLM
        messages = [{"role": "user", "content": prompt}]
        response_text = ""

        for chunk in self.llm.think(messages, **kwargs):
            if chunk:
                response_text += chunk

        # 解析输出
        result = self._parse_result(response_text)
        return result

    def _reflection_phase(self, task: str, **kwargs) -> Iterator[str]:
        """
        反思阶段

        Args:
            task: 任务描述
            **kwargs: LLM 调用参数

        Yields:
            反思过程的输出

        Returns:
            反思结果
        """
        last_result = self.memory.get_last_execution()

        prompt = self.reflect_prompt.format(
            task=task,
            content=last_result
        )

        # 调用 LLM
        messages = [{"role": "user", "content": prompt}]
        response_text = ""

        for chunk in self.llm.think(messages, **kwargs):
            if chunk:
                response_text += chunk
                yield chunk

        return response_text

    def _parse_result(self, text: str) -> str:
        """
        解析 LLM 输出，提取最终结果

        Args:
            text: LLM 返回的文本

        Returns:
            提取的结果
        """
        # 查找 Finish 标记
        finish_match = re.search(r"Finish\[(.*?)\]", text, re.DOTALL)
        if finish_match:
            return finish_match.group(1).strip()

        # 如果没有 Finish，返回整个文本
        return text.strip()

    def _is_satisfactory(self, reflection: str) -> bool:
        """
        判断反思结果是否表示满意

        Args:
            reflection: 反思结果

        Returns:
            是否满意
        """
        satisfactory_keywords = ["无需改进", "no need", "已经很好", "already good", "满意"]
        reflection_lower = reflection.lower()

        for keyword in satisfactory_keywords:
            if keyword.lower() in reflection_lower:
                return True

        return False
