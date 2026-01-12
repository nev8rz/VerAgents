"""Planner-Solver Agent 实现 - 规划与求解分离的智能体"""

import re
from typing import Optional, List, Iterator
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


class Planner:
    """规划器 - 负责将复杂问题分解为可执行的步骤列表"""

    def __init__(
        self,
        llm: VerAgentLLM,
        tool_registry: ToolRegistry,
        prompt_template: Optional[str] = None
    ):
        """
        初始化规划器

        Args:
            llm: LLM 实例
            tool_registry: 工具注册表
            prompt_template: 自定义提示词模板
        """
        self.llm = llm
        self.tool_registry = tool_registry
        self.prompt_template = prompt_template if prompt_template else DEFAULT_PLANNER_PROMPT

    def plan(self, question: str, **kwargs) -> List[str]:
        """
        生成执行计划

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
        return self._parse_plan(response_text)

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
Action: 选择合适的工具获取信息，格式为：
- `{{tool_name}}[{{参数}}]`：调用工具获取信息
- `Finish[结论]`：当当前步骤完成时

## ⚠️ 重要提醒
1. 专注于完成当前步骤，不要试图跳到下一步
2. 如果当前步骤需要多次工具调用，继续调用直到完成
3. 使用简洁的语言总结当前步骤的结果
4. 完成当前步骤后使用 Finish

现在开始执行当前步骤："""


class Solver:
    """求解器 - 负责执行单个任务步骤（使用 ReAct 模式）"""

    def __init__(
        self,
        llm: VerAgentLLM,
        tool_registry: ToolRegistry,
        prompt_template: Optional[str] = None,
        max_iterations: int = 5
    ):
        """
        初始化求解器

        Args:
            llm: LLM 实例
            tool_registry: 工具注册表
            prompt_template: 自定义提示词模板
            max_iterations: 每个步骤的最大迭代次数
        """
        self.llm = llm
        self.tool_registry = tool_registry
        self.prompt_template = prompt_template if prompt_template else DEFAULT_SOLVER_PROMPT
        self.max_iterations = max_iterations

    def solve(
        self,
        _question: str,
        plan: List[str],
        step_index: int,
        step_results: List[str],
        **kwargs
    ) -> Iterator[str]:
        """
        执行单个步骤

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

        tools_desc = self.tool_registry.get_tools_description()
        plan_str = "\n".join([f"{i}. {s}" for i, s in enumerate(plan, 1)])
        previous_results_str = "\n".join([
            f"步骤 {i}: {result}" for i, result in enumerate(step_results, 1)
        ]) if step_results else "（暂无已完成步骤）"

        prompt = self.prompt_template.format(
            tools=tools_desc,
            plan=plan_str,
            current_step=current_step,
            step_index=step_index,
            total_steps=len(plan),
            previous_results=previous_results_str
        )

        messages = [{"role": "user", "content": prompt}]
        current_step_history = []

        for _ in range(self.max_iterations):
            # 静默收集 LLM 输出
            response_text = ""
            for chunk in self.llm.think(messages, **kwargs):
                if chunk:
                    response_text += chunk

            # 解析输出
            thought, action = self._parse_output(response_text)

            if thought:
                yield f"🤔 思考: {thought}\n"

            if not action:
                yield "⚠️ 警告：未能解析出有效的 Action。\n"
                break

            # 检查是否完成当前步骤
            if action.startswith("Finish"):
                step_result = self._parse_action_input(action)
                yield f"✅ 步骤完成: {step_result}\n"
                return step_result

            # 执行工具调用
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or tool_input is None:
                yield "⚠️ 无效的 Action 格式\n"
                continue

            yield f"🎬 行动: {tool_name}[{tool_input}]\n"

            # 调用工具
            tool_args = self._build_tool_args(tool_name, tool_input)
            observation = self.tool_registry.execute(tool_name, tool_args)

            yield f"👀 观察: {observation}\n"

            # 更新历史
            current_step_history.append(f"Action: {action}")
            current_step_history.append(f"Observation: {observation}")

        # 如果循环结束但没有 Finish
        yield "⏰ 当前步骤已达到最大迭代次数。\n"
        return f"步骤 {step_index} 执行未完成，可能需要更多信息。"

    def _parse_output(self, text: str) -> tuple[Optional[str], Optional[str]]:
        """解析 LLM 输出，提取思考和行动"""
        thought_match = re.search(r"Thought:\s*(.*?)(?:\n|$)", text)
        thought = thought_match.group(1).strip() if thought_match else None

        action_match = re.search(r"Action:\s*(.*?)(?:\n|$)", text)

        if not action_match:
            direct_match = re.search(r"(?:^|\n)\s*(\w+\[.*?\])(?:\s|$)", text, re.MULTILINE)
            if direct_match:
                action = direct_match.group(1).strip()
            else:
                bracket_match = re.search(r"(\w+\[.*?\])", text)
                action = bracket_match.group(1).strip() if bracket_match else None
        else:
            action = action_match.group(1).strip()

        return thought, action

    def _parse_action(self, action_text: str) -> tuple[Optional[str], Optional[str]]:
        """解析行动文本，提取工具名称和输入"""
        match = re.match(r"(\w+)\[(.*)\]", action_text)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def _parse_action_input(self, action_text: str) -> str:
        """解析行动输入（用于 Finish）"""
        match = re.match(r"\w+\[(.*)\]", action_text)
        return match.group(1) if match else ""

    def _build_tool_args(self, tool_name: str, tool_input: str) -> dict:
        """智能构建工具参数"""
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return {"query": tool_input}

        schema = tool.openai_schema.get("function", {})
        params = schema.get("parameters", {})
        is_toolkit = "discriminator" in params

        # 处理 JSON 格式
        import json
        tool_input_stripped = tool_input.strip()
        if tool_input_stripped.startswith("{") and tool_input_stripped.endswith("}"):
            try:
                json_args = json.loads(tool_input_stripped)
                if isinstance(json_args, dict):
                    return json_args
            except json.JSONDecodeError:
                pass

        # 解析 key=value 格式
        if "=" in tool_input:
            result = {}
            parts = tool_input.split(",")
            for part in parts:
                part = part.strip()
                if "=" in part:
                    key, value = part.split("=", 1)
                    result[key.strip()] = value.strip()
            return result
        else:
            # 简单格式
            if is_toolkit:
                discriminator = params.get("discriminator", {})
                mapping = discriminator.get("mapping", {})
                actions = list(mapping.keys())
                default_action = actions[0] if actions else "action"

                defs = params.get("$defs", {})
                if defs:
                    first_def_key = list(defs.keys())[0]
                    first_def = defs[first_def_key]
                    props = first_def.get("properties", {})
                    if props:
                        param_names = [k for k in props.keys() if k != "action"]
                        if param_names:
                            return {"action": default_action, param_names[0]: tool_input.strip()}

                return {"action": default_action, "input": tool_input.strip()}
            else:
                properties = params.get("properties", {})
                if properties:
                    first_param = list(properties.keys())[0]
                    return {first_param: tool_input.strip()}
                else:
                    return {"query": tool_input.strip()}


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

        # 创建独立的 Planner 和 Solver
        self.planner = Planner(llm, self.tool_registry, custom_planner_prompt)
        self.solver = Solver(llm, self.tool_registry, custom_solver_prompt, max_steps_per_task)

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
            final_answer = chunk
        return final_answer

    def run_stream(self, input_text: str, **kwargs) -> Iterator[str]:
        """运行 Planner-Solver Agent（流式输出）"""
        # 重置内部状态
        self.plan = []
        self.step_results = []

        if self.verbose:
            yield f"\n🤖 {self.name} 开始处理问题: {input_text}\n"

        # ============ 阶段 1: 规划 ============
        if self.verbose:
            yield "\n" + "="*50 + "\n"
            yield "📋 阶段 1: 规划\n"
            yield "="*50 + "\n"
            yield "🧠 正在分析问题并制定计划...\n"

        self.plan = self.planner.plan(input_text, **kwargs)

        if not self.plan:
            if self.verbose:
                yield "\n❌ 规划失败，无法生成执行计划。\n"
            return

        if self.verbose:
            yield f"\n✅ 规划完成，共 {len(self.plan)} 个步骤：\n"
            for i, step in enumerate(self.plan, 1):
                yield f"   {i}. {step}\n"

        # ============ 阶段 2: 求解 ============
        if self.verbose:
            yield "\n" + "="*50 + "\n"
            yield "🔧 阶段 2: 执行计划\n"
            yield "="*50 + "\n"

        for i, step in enumerate(self.plan, 1):
            if self.verbose:
                yield f"\n--- 步骤 {i}/{len(self.plan)} ---\n"
                yield f"📌 任务: {step}\n"

            # Solver 返回的是生成器，需要迭代
            result = None
            for chunk in self.solver.solve(input_text, self.plan, i, self.step_results, **kwargs):
                yield chunk
                # 提取最终结果（包含 "✅ 步骤完成:" 的行）
                if chunk.startswith("✅ 步骤完成:"):
                    result = chunk.replace("✅ 步骤完成:", "").strip()

            if result:
                self.step_results.append(result)

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
