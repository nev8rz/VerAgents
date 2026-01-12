"""Reflection Agent 实现 - 自我反思与迭代优化的智能体

Reflection Agent 是一个元智能体，它可以包装其他执行器（如 ReActAgent 或 PlannerSolverAgent），
通过自我反思和迭代改进来提升输出质量。
"""

from typing import Optional, Iterator, Union
from ..core.agent import Agent
from ..core.llm import VerAgentLLM
from ..core.message import Message
from ..tools.registry import ToolRegistry


# 默认反思提示词模板
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

请分析这个回答的质量。

## 输出要求
- 如果回答已经完美，请只回答："无需改进"
- 如果需要改进，请简洁列出主要问题和建议

现在开始评审："""


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


class ReflectionAgent:
    """
    Reflection Agent - 自我反思与迭代优化的智能体

    基于 Reflexion 论文的实现，通过自我反思和迭代改进来提升输出质量。

    核心思想：
    1. 执行初始任务（使用指定的执行器：ReActAgent 或 PlannerSolverAgent）
    2. 对结果进行自我反思
    3. 根据反思结果进行优化
    4. 迭代改进直到满意或达到最大迭代次数

    工作流程：
        Initial(Executor) → Reflect → Refine(Executor) → Reflect → Refine(Executor) → ... → Final Answer

    支持的执行器：
    - ReActAgent: 推理与行动结合的智能体
    - PlannerSolverAgent: 规划与求解分离的智能体

    特别适合：
    - 代码生成与优化
    - 文档写作
    - 分析报告
    - 需要多轮迭代完善的任务
    """

    def __init__(
        self,
        name: str,
        executor: Agent,
        llm: VerAgentLLM,
        reflect_prompt: Optional[str] = None,
        max_iterations: int = 3,
        verbose: bool = True
    ):
        """
        初始化 ReflectionAgent

        Args:
            name: Agent 名称
            executor: 执行器实例（ReActAgent 或 PlannerSolverAgent）
            llm: LLM 实例（用于反思阶段）
            reflect_prompt: 自定义反思提示词
            max_iterations: 最大反思迭代次数
            verbose: 是否显示详细过程
        """
        self.name = name
        self.executor = executor
        self.llm = llm
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.memory = SimpleMemory()

        # 设置反思提示词
        self.reflect_prompt = reflect_prompt if reflect_prompt else DEFAULT_REFLECT_PROMPT

    def add_tool(self, tool):
        """添加工具到执行器的工具注册表"""
        if hasattr(self.executor, 'add_tool'):
            self.executor.add_tool(tool)
        elif hasattr(self.executor, 'tool_registry'):
            self.executor.tool_registry.register(tool)

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
        seen_results = set()
        for chunk in self.run_stream(input_text, **kwargs):
            # 提取非装饰性的内容作为最终答案
            if chunk and not chunk.startswith("\n") and not chunk.startswith("=") and not chunk.startswith("-") and not chunk.startswith("🤖") and not chunk.startswith("📝") and not chunk.startswith("🔍") and not chunk.startswith("🎉") and not chunk.startswith("→") and not chunk.startswith("✅") and not chunk.startswith("🔄") and not chunk.startswith("📊"):
                if chunk not in seen_results:
                    final_answer = chunk
                    seen_results.add(chunk)
        return final_answer if final_answer else "未能生成有效结果"

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

        # 使用执行器进行初始执行（流式输出）
        initial_result = yield from self._run_executor_stream(input_text, **kwargs)
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

            # c. 优化 - 使用执行器重新执行（流式输出）
            if self.verbose:
                yield "\n→ 正在进行优化...\n"

            refined_result = yield from self._run_executor_stream(input_text, **kwargs)
            self.memory.add_record("execution", refined_result)

            if self.verbose:
                yield f"\n🔄 优化完成\n"

        # ============ 阶段 3: 返回最终结果 ============
        final_result = self.memory.get_last_execution()

        if not final_result:
            final_result = "抱歉，无法生成有效结果。"

        if self.verbose:
            yield "\n" + "="*50 + "\n"
            yield "🎉 最终结果\n"
            yield "="*50 + "\n"
            yield final_result

        # 保存到历史记录
        if hasattr(self.executor, 'add_message'):
            self.executor.add_message(Message(input_text, "user"))
            self.executor.add_message(Message(final_result, "assistant"))

    def run(self, input_text: str, **kwargs) -> str:
        """
        运行 Reflection Agent（非流式，返回最终答案）

        Args:
            input_text: 任务描述
            **kwargs: 其他参数

        Returns:
            最终优化后的结果
        """
        # 直接从 memory 中获取最后执行的结果
        # 先运行流式版本完成所有处理
        for _ in self.run_stream(input_text, **kwargs):
            pass
        # 然后从记忆中获取最终结果
        final_result = self.memory.get_last_execution()
        return final_result if final_result else "未能生成有效结果"

    def _run_executor_stream(self, task: str, **kwargs):
        """
        使用执行器运行任务（流式）

        Args:
            task: 任务描述
            **kwargs: 其他参数

        Yields:
            执行器的流式输出

        Returns:
            执行结果
        """
        # 优先使用 run_stream 获取详细输出
        if hasattr(self.executor, 'run_stream'):
            result = ""
            for chunk in self.executor.run_stream(task, **kwargs):
                if chunk:
                    result = chunk
                    yield chunk
            return result
        elif hasattr(self.executor, 'run'):
            result = self.executor.run(task, **kwargs)
            yield result
            return result
        else:
            raise AttributeError(f"执行器 {type(self.executor).__name__} 没有 run 或 run_stream 方法")

    def _run_executor(self, task: str, **kwargs) -> str:
        """
        使用执行器运行任务（非流式，向后兼容）

        Args:
            task: 任务描述
            **kwargs: 其他参数

        Returns:
            执行结果
        """
        # 优先使用 run_stream 获取详细输出
        if hasattr(self.executor, 'run_stream'):
            result = ""
            for chunk in self.executor.run_stream(task, **kwargs):
                if chunk:
                    result = chunk
            return result
        elif hasattr(self.executor, 'run'):
            return self.executor.run(task, **kwargs)
        else:
            raise AttributeError(f"执行器 {type(self.executor).__name__} 没有 run 或 run_stream 方法")

    def _reflection_phase(self, task: str, **kwargs):
        """
        反思阶段

        Args:
            task: 任务描述
            **kwargs: LLM 调用参数

        Yields:
            反思过程的输出（LLM 流式响应）

        Returns:
            反思结果的完整文本
        """
        last_result = self.memory.get_last_execution()

        prompt = self.reflect_prompt.format(
            task=task,
            content=last_result
        )

        # 调用 LLM（静默模式，不输出 "thinking..." 消息）
        messages = [{"role": "user", "content": prompt}]
        response_text = ""

        for chunk in self.llm.think(messages, quiet=True, **kwargs):
            if chunk:
                response_text += chunk
                yield chunk

        return response_text.strip()

    def _is_satisfactory(self, reflection: str) -> bool:
        """判断反思结果是否表示满意"""
        satisfactory_keywords = ["无需改进", "no need", "已经很好", "already good", "满意", "完美"]
        reflection_lower = reflection.lower()

        for keyword in satisfactory_keywords:
            if keyword.lower() in reflection_lower:
                return True

        return False
