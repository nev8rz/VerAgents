"""Planner-Solver Agent 实现 - 规划与求解分离的智能体

重构说明:
- Planner 现在继承自 Agent，成为独立的规划智能体
- Solver 继承自 ReActAgent，复用工具解析和执行逻辑
"""

import re
from typing import Optional, List, Iterator
from .react import ReActAgent
from ..core.agent import Agent
from ..core.llm import VerAgentLLM
from ..core.config import Config
from ..core.message import Message
from ..tools.registry import ToolRegistry


# 默认规划器提示词模板
DEFAULT_PLANNER_PROMPT = """你是一个专业的任务规划专家。你的职责是将复杂问题分解为清晰的执行步骤。

## 可用工具
{tools}

## 工作流程
1. 仔细分析用户的问题
2. 识别需要哪些信息
3. 确定步骤的合理顺序
4. 输出详细的执行计划

## 输出格式
请严格按照以下格式输出：

Plan:
1. [第一步描述]
2. [第二步描述]
3. [第三步描述]
...

## ⚠️ 重要提醒
1. 每个步骤应该是具体可执行的
2. 步骤之间应该有逻辑顺序
3. 考虑哪些步骤可以合并
4. 通常 3-5 个步骤比较合适
5. 步骤描述要简洁明了

## 当前任务
**Question:** {question}

现在开始规划："""


class Planner(Agent):
    """规划器 Agent - 负责将复杂问题分解为可执行的步骤列表

    继承自 Agent，作为独立的规划智能体运行。
    """

    def __init__(
        self,
        name: str,
        llm: VerAgentLLM,
        tool_registry: ToolRegistry,
        prompt_template: Optional[str] = None,
        config: Optional[Config] = None
    ):
        """
        初始化规划器 Agent

        Args:
            name: Agent 名称
            llm: LLM 实例
            tool_registry: 工具注册表
            prompt_template: 自定义提示词模板
            config: 配置对象
        """
        super().__init__(name, llm, system_prompt=None, config=config)
        self.tool_registry = tool_registry
        self.prompt_template = prompt_template if prompt_template else DEFAULT_PLANNER_PROMPT

    def run(self, question: str, **kwargs) -> List[str]:
        """
        生成执行计划（非流式）

        Args:
            question: 用户问题
            **kwargs: LLM 调用参数

        Returns:
            步骤列表
        """
        tools_desc = self.tool_registry.get_tools_description()
        prompt = self.prompt_template.format(
            tools=tools_desc,
            question=question
        )

        messages = [{"role": "user", "content": prompt}]

        # 收集 LLM 响应
        response_text = ""
        for chunk in self.llm.think(messages, **kwargs):
            response_text += chunk

        # 解析计划
        steps = self._parse_plan(response_text)

        # 保存到历史记录
        self.add_message(Message(question, "user"))
        self.add_message(Message(f"Plan:\n" + "\n".join([f"{i}. {s}" for i, s in enumerate(steps, 1)]), "assistant"))

        return steps

    def run_stream(self, question: str, **kwargs) -> Iterator[str]:
        """流式运行规划器"""
        tools_desc = self.tool_registry.get_tools_description()
        prompt = self.prompt_template.format(
            tools=tools_desc,
            question=question
        )

        messages = [{"role": "user", "content": prompt}]

        yield f"\n🧠 {self.name} 正在分析问题并制定计划..."

        response_text = ""
        for chunk in self.llm.think(messages, **kwargs):
            if chunk:
                response_text += chunk
                yield chunk

        steps = self._parse_plan(response_text)

        # 保存到历史记录
        self.add_message(Message(question, "user"))
        self.add_message(Message(f"Plan:\n" + "\n".join([f"{i}. {s}" for i, s in enumerate(steps, 1)]), "assistant"))

        yield f"\n\n✅ 规划完成，共 {len(steps)} 个步骤：\n"
        for i, step in enumerate(steps, 1):
            yield f"   {i}. {step}\n"

    def _parse_plan(self, text: str) -> List[str]:
        """
        解析 LLM 输出，提取执行计划

        Args:
            text: LLM 返回的文本

        Returns:
            步骤列表
        """
        # 查找 Plan: 标记后的内容
        plan_match = re.search(r"Plan:\s*(.*?)(?:\n\n|\n(?=[A-Z])|$)", text, re.DOTALL | re.IGNORECASE)

        if plan_match:
            plan_text = plan_match.group(1).strip()
        else:
            # 尝试直接提取所有编号列表
            plan_text = text.strip()

        # 解析编号列表
        steps = []
        lines = plan_text.split("\n")

        for line in lines:
            line = line.strip()
            # 匹配 "1." 或 "1、" 格式
            match = re.match(r"^[\d]+[\.\、]\s*(.+)", line)
            if match:
                step = match.group(1).strip()
                # 清理可能的 markdown 标记
                step = re.sub(r'^[\*\-\+]+\s*', '', step)
                if step:
                    steps.append(step)

        return steps


# 默认求解器提示词模板
DEFAULT_SOLVER_PROMPT = """你是一个任务执行专家。你需要按照给定的计划，一步步完成任务。

## 可用工具
{tools}

## 执行计划
{plan}

## 当前步骤
当前执行: {current_step}
进度: {step_index} / {total_steps}

## 已完成步骤的结果
{previous_results}

## 工作流程
请严格按照以下格式进行回应，专注于完成当前步骤：

Thought: 分析当前步骤，确定需要什么信息或如何完成任务
Action: 选择合适的工具获取信息：
- 调用工具：`工具名[参数]` 或 `工具名[参数名=值]`
- 完成步骤：`Finish[结论]`

## ⚠️ 重要提醒
1. 每次回应必须包含 Thought 和 Action 两部分
2. 工具调用格式严格遵循：工具名[参数]
3. **对于工具集（Toolkit），必须指定 action 参数！**
   - 例如：`WeatherFetcher[action=get_weather, location=北京]`
   - action 参数必须使用工具说明中列出的可用操作名称
4. 对于多参数工具，使用 `参数名=值` 格式
5. 专注于完成当前步骤，不要试图跳到下一步
6. 如果当前步骤需要多次工具调用，继续调用直到完成
7. 使用简洁的语言总结当前步骤的结果
8. 完成当前步骤后使用 Finish

## 执行历史
{history}

现在开始执行当前步骤："""


class Solver(ReActAgent):
    """求解器 Agent - 负责执行单个任务步骤

    继承自 ReActAgent，复用 Thought-Action-Observation 循环逻辑。
    与标准 ReActAgent 的区别在于：
    - 专注于单步执行而非全局任务
    - 接收计划上下文（完整计划、当前步骤、前置结果）
    - 使用 Finish 标记当前步骤完成
    """

    def __init__(
        self,
        name: str,
        llm: VerAgentLLM,
        tool_registry: ToolRegistry,
        prompt_template: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        verbose: bool = True
    ):
        """
        初始化求解器 Agent

        Args:
            name: Agent 名称
            llm: LLM 实例
            tool_registry: 工具注册表
            prompt_template: 自定义提示词模板
            config: 配置对象
            max_steps: 每个步骤的最大迭代次数
            verbose: 是否显示详细过程
        """
        # 使用自定义的 solver 提示词初始化 ReActAgent
        super().__init__(
            name=name,
            llm=llm,
            tool_registry=tool_registry,
            system_prompt=None,
            config=config,
            max_steps=max_steps,
            custom_prompt=prompt_template if prompt_template else DEFAULT_SOLVER_PROMPT,
            verbose=verbose
        )

    def run(
        self,
        question: str,
        plan: List[str],
        step_index: int,
        step_results: List[str],
        **kwargs
    ) -> str:
        """
        执行单个步骤（非流式）

        Args:
            question: 原始问题
            plan: 完整计划
            step_index: 当前步骤索引（从 1 开始）
            step_results: 已完成步骤的结果
            **kwargs: LLM 调用参数

        Returns:
            步骤执行结果
        """
        final_result = ""
        for chunk in self.run_stream(question, plan, step_index, step_results, **kwargs):
            if "✅ 步骤完成:" in chunk:
                final_result = chunk.replace("✅ 步骤完成:", "").strip()
        return final_result

    def run_stream(
        self,
        question: str,
        plan: List[str],
        step_index: int,
        step_results: List[str],
        **kwargs
    ) -> Iterator[str]:
        """
        执行单个步骤（流式输出）

        Args:
            question: 原始问题
            plan: 完整计划
            step_index: 当前步骤索引（从 1 开始）
            step_results: 已完成步骤的结果
            **kwargs: LLM 调用参数

        Yields:
            执行过程的输出
        """
        current_step = plan[step_index - 1]

        # 初始化执行上下文
        self.current_history = []
        current_step_count = 0

        if self.verbose:
            yield f"\n📌 任务: {current_step}\n"

        # 执行 ReAct 循环
        while current_step_count < self.max_steps:
            current_step_count += 1

            if self.verbose:
                yield f"  ── 迭代 {current_step_count} ──\n"

            # 构建提示词
            tools_desc = self.tool_registry.get_tools_description()
            plan_str = "\n".join([f"{i}. {s}" for i, s in enumerate(plan, 1)])
            previous_results_str = "\n".join([
                f"步骤 {i}: {result}" for i, result in enumerate(step_results, 1)
            ]) if step_results else "（暂无已完成步骤）"
            history_str = "\n".join(self.current_history) if self.current_history else "（暂无执行历史）"

            prompt = self.prompt_template.format(
                tools=tools_desc,
                plan=plan_str,
                current_step=current_step,
                step_index=step_index,
                total_steps=len(plan),
                previous_results=previous_results_str,
                history=history_str
            )

            # 调用 LLM
            messages = [{"role": "user", "content": prompt}]
            response_text = ""
            for chunk in self.llm.think(messages, **kwargs):
                if chunk:
                    response_text += chunk

            # 解析输出（复用 ReActAgent 的方法）
            thought, action = self._parse_output(response_text)

            if thought and self.verbose:
                yield f"  🤔 思考: {thought}\n"

            if not action:
                yield "  ⚠️ 警告：未能解析出有效的 Action。\n"
                break

            # 检查是否完成当前步骤
            if action.startswith("Finish"):
                step_result = self._parse_action_input(action)
                if self.verbose:
                    yield f"  ✅ 步骤完成: {step_result}\n"
                return

            # 执行工具调用（复用 ReActAgent 的方法）
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or tool_input is None:
                yield "  ⚠️ 无效的 Action 格式\n"
                continue

            if self.verbose:
                yield f"  🎬 行动: {tool_name}[{tool_input}]\n"

            # 调用工具（复用 ReActAgent 的方法）
            tool_args = self._build_tool_args(tool_name, tool_input)
            observation = self.tool_registry.execute(tool_name, tool_args)

            if self.verbose:
                yield f"  👀 观察: {observation}\n"

            # 更新历史
            self.current_history.append(f"Action: {action}")
            self.current_history.append(f"Observation: {observation}")

        # 达到最大迭代次数
        if self.verbose:
            yield "  ⏰ 当前步骤已达到最大迭代次数。\n"


class PlannerSolverAgent(Agent):
    """
    Planner-Solver Agent - 规划与求解分离的智能体

    采用规划-求解分离架构的智能体，能够：
    1. 在规划阶段分析问题并制定执行计划
    2. 在求解阶段依次执行每个子任务
    3. 每个子任务使用 ReAct 模式调用工具
    4. 汇总各步骤结果给出最终答案

    特别适合需要结构化分解的复杂任务。
    """

    def __init__(
        self,
        name: str,
        llm: VerAgentLLM,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps_per_task: int = 5,
        custom_planner_prompt: Optional[str] = None,
        custom_solver_prompt: Optional[str] = None,
        verbose: bool = True
    ):
        """
        初始化 PlannerSolverAgent

        Args:
            name: Agent 名称
            llm: LLM 实例
            tool_registry: 工具注册表（可选，如果不提供则创建空的）
            system_prompt: 系统提示词
            config: 配置对象
            max_steps_per_task: 每个子任务的最大执行步数
            custom_planner_prompt: 自定义规划器提示词模板
            custom_solver_prompt: 自定义求解器提示词模板
            verbose: 是否显示详细过程
        """
        super().__init__(name, llm, system_prompt, config)

        # 如果没有提供 tool_registry，创建一个空的
        if tool_registry is None:
            self.tool_registry = ToolRegistry()
        else:
            self.tool_registry = tool_registry

        self.max_steps_per_task = max_steps_per_task
        self.verbose = verbose

        # 创建独立的 Planner 和 Solver Agent
        self.planner = Planner(
            name=f"{name}_Planner",
            llm=llm,
            tool_registry=self.tool_registry,
            prompt_template=custom_planner_prompt,
            config=config
        )

        self.solver = Solver(
            name=f"{name}_Solver",
            llm=llm,
            tool_registry=self.tool_registry,
            prompt_template=custom_solver_prompt,
            config=config,
            max_steps=max_steps_per_task,
            verbose=verbose
        )

        # 内部状态
        self.plan: List[str] = []
        self.step_results: List[str] = []

    def add_tool(self, tool):
        """添加工具到工具注册表"""
        self.tool_registry.register(tool)

    def run(self, input_text: str, **kwargs) -> str:
        """运行 Planner-Solver Agent（非流式，返回最终答案）"""
        final_answer = ""
        for chunk in self.run_stream(input_text, **kwargs):
            if chunk:
                # 显示输出（与 run_stream 行为一致）
                print(chunk, end="", flush=True)
                # 收集最终答案（跳过装饰性内容）
                if not chunk.startswith("\n") and not chunk.startswith("=") and not chunk.startswith("📋") and not chunk.startswith("🔧") and not chunk.startswith("📊"):
                    final_answer = chunk
        print()  # 换行
        return final_answer

    def run_stream(self, input_text: str, **kwargs) -> Iterator[str]:
        """运行 Planner-Solver Agent（流式输出）"""
        # 重置内部状态
        self.plan = []
        self.step_results = []

        if self.verbose:
            yield f"\n🤖 {self.name} 开始处理问题: {input_text}"

        # ============ 阶段 1: 规划 ============
        if self.verbose:
            yield "\n" + "="*50 + "\n"
            yield "📋 阶段 1: 规划\n"
            yield "="*50 + "\n"

        # 使用 Planner Agent 的流式输出
        plan_output = ""
        for chunk in self.planner.run_stream(input_text, **kwargs):
            yield chunk
            plan_output += chunk

        self.plan = self.planner.run(input_text, **kwargs)

        if not self.plan:
            if self.verbose:
                yield "\n❌ 规划失败，无法生成执行计划。\n"
            return

        # ============ 阶段 2: 求解 ============
        if self.verbose:
            yield "\n" + "="*50 + "\n"
            yield "🔧 阶段 2: 执行计划\n"
            yield "="*50 + "\n"

        for i, step in enumerate(self.plan, 1):
            if self.verbose:
                yield f"\n--- 步骤 {i}/{len(self.plan)} ---"

            # Solver 返回的是生成器，需要迭代
            step_result = None
            for chunk in self.solver.run_stream(input_text, self.plan, i, self.step_results, **kwargs):
                yield chunk
                # 提取最终结果（包含 "✅ 步骤完成:" 的行）
                if "✅ 步骤完成:" in chunk:
                    step_result = chunk.replace("✅ 步骤完成:", "").strip()

            if step_result:
                self.step_results.append(step_result)

        # ============ 阶段 3: 汇总 ============
        if self.verbose:
            yield "\n" + "="*50 + "\n"
            yield "📊 阶段 3: 结果汇总\n"
            yield "="*50 + "\n"

        yield from self._summary_phase(input_text)

    def _summary_phase(self, original_question: str) -> Iterator[str]:
        """汇总阶段：总结所有步骤的结果"""
        if not self.step_results:
            yield "\n⚠️ 没有执行结果可以汇总。\n"
            return

        # 构建汇总提示词
        summary_prompt = f"""基于以下执行步骤的结果，请对原始问题给出一个综合、准确的最终答案。

## 原始问题
{original_question}

## 执行计划与结果
"""

        for i, (step, result) in enumerate(zip(self.plan, self.step_results), 1):
            summary_prompt += f"\n步骤 {i}: {step}\n结果: {result}\n"

        summary_prompt += """
## 要求
请基于以上信息，给出一个简洁、准确的最终答案。如果某些步骤的结果不完整或不相关，请在答案中说明。

## 最终答案："""

        messages = [{"role": "user", "content": summary_prompt}]

        if self.verbose:
            yield "\n🔄 正在汇总结果...\n"

        final_answer = ""
        for chunk in self.llm.think(messages):
            if chunk:
                final_answer += chunk
                yield chunk

        # 保存到历史记录
        self.add_message(Message(original_question, "user"))
        self.add_message(Message(final_answer, "assistant"))
