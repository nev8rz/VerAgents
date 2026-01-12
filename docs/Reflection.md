# Reflection Agent（反思智能体）

## 一、论文基本信息

**论文标题**

> *Reflexion: Language Agents with Verbal Reinforcement Learning*

**作者**

Shinn N., Cassano F., Gopinath A., Narasimhan K.（Princeton / Google DeepMind）

**首次提出时间**

2023 年

**核心关键词**

* Reflection（反思）
* Iterative Refinement（迭代优化）
* Verbal Reinforcement Learning（语言强化学习）
* Self-Evaluation（自我评估）

---

## 二、研究动机（为什么要 Reflection）

### 问题：LLM 的一次性输出限制

传统的 LLM 应用模式：

```
用户提问 → LLM → 直接回答
```

**问题：**
* 一次生成，没有自我检查机会
* 复杂任务容易出现错误
* 无法从错误中学习改进
* 代码类任务容易有小错误

---

### ✅ Reflection 的核心动机

> **让 LLM 在输出后进行自我反思，根据反思结果进行迭代改进，直到达到满意的质量。**

---

## 三、Reflection 的核心思想（一句话）

> **初始输出 → 自我反思 → 迭代改进 → 再次反思 → ... → 最终答案**

这是 Reflection 的灵魂。

---

## 四、Reflection 的整体框架

### 交互格式（标准）

```
┌─────────────────────────────────────────────────────────────┐
│                      用户提出任务                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     第 1 轮：初始执行                          │
│  Actor: 生成初始回答                                          │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     第 1 轮：自我反思                          │
│  Evaluator: 评估回答质量，找出问题，提出改进建议               │
└─────────────────────────────────────────────────────────────┘
                          │
                    满意吗？
                    │       │
                   否       是
                    │       │
                    ▼       ▼
┌───────────────────────┐  ┌───────────────────────┐
│   第 N+1 轮：迭代改进   │  │     返回最终答案       │
│   Actor: 根据反馈改进   │  │                      │
└───────────────────────┘  └───────────────────────┘
```

---

### 关键组成部分

| 组件           | 作用                     |
| -------------- | ------------------------ |
| **Actor**      | 执行任务，生成回答         |
| **Evaluator**  | 反思评估，找出问题         |
| **Memory**     | 存储历史轨迹，提供上下文    |

---

## 五、Reflection 的工作流程（详细）

### 场景：代码生成任务

**任务：** "编写一个 Python 函数计算斐波那契数列"

---

### 第 1 轮：初始执行

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

---

### 第 1 轮：反思

**Evaluator 反馈：**
> - 问题：函数没有输入验证
> - 问题：递归实现效率低，大数会栈溢出
> - 建议：添加边界检查，改用迭代实现

---

### 第 2 轮：迭代改进

```python
def fibonacci(n):
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```

---

### 第 2 轮：反思

**Evaluator 反馈：**
> 无需改进

---

### 最终答案

返回改进后的代码。

---

## 六、Reflection 与其他 Agent 的对比

| 对比维度        | ReAct              | Planner-Solver      | Reflection              |
| -------------- | ------------------ | ------------------- | ----------------------- |
| **核心模式**      | Thought-Action     | Plan-Execute        | Act-Reflect-Refine     |
| **迭代方式**      | 步骤级迭代           | 任务级分解            | 轮次级优化               |
| **自我纠错**      | 通过工具观察        | 通过步骤反馈         | 通过自我反思            |
| **适用场景**      | 需要外部信息        | 复杂多步骤任务       | 需要质量提升的任务       |
| **输出质量**      | 取决于推理能力      | 取决于规划质量       | **持续提升**            |

---

## 七、Reflection 的优势

### ✅ 1. 显著提升输出质量

通过多轮反思和改进，逐步逼近最优解。

### ✅ 2. 自动错误修复

能够发现并修复代码、文本中的错误。

### ✅ 3. 无需额外标注

使用同一个 LLM，不需要额外的训练数据。

### ✅ 4. 灵活的迭代次数

可以根据任务复杂度调整最大迭代次数。

---

## 八、Reflection 的 Prompt 设计

### 1. 初始执行 Prompt

```python
INITIAL_PROMPT = """你是一个专业的任务执行助手。请根据以下要求完成任务。

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
```

### 2. 反思 Prompt

```python
REFLECT_PROMPT = """你是一个严谨的质量评审员。请仔细审查以下回答，并找出可能的问题或改进空间。

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
```

### 3. 优化 Prompt

```python
REFINE_PROMPT = """你是一个专业的任务执行助手。请根据反馈意见改进你的回答。

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
```

---

## 九、典型应用场景

### 场景 1：代码生成与优化

```python
# 任务：实现一个二分查找算法

# 第 1 轮：可能有边界问题
def binary_search(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid
        else:
            right = mid
    return -1

# 反思：边界条件错误
# 第 2 轮：修复后的代码
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

### 场景 2：文档写作

```python
# 任务：编写 API 文档

# 第 1 轮：可能缺少示例
"""
# API: get_user
获取用户信息
参数：user_id
返回：用户对象
"""

# 反思：缺少使用示例
# 第 2 轮：补充完整文档
"""
# API: get_user
获取用户信息

## 参数
- user_id (int): 用户ID

## 返回
- User: 用户对象，包含 id, name, email 等字段

## 示例
user = get_user(123)
print(user.name)
"""
```

### 场景 3：数学问题求解

```python
# 任务：求解微积分问题

# 第 1 轮：可能有计算错误
"∫x²dx = x³/3"

# 反思：缺少常数项
# 第 2 轮：修正答案
"∫x²dx = x³/3 + C"
```

---

## 十、实现要点

### 1. 简单记忆模块

```python
class SimpleMemory:
    def __init__(self):
        self.records: list[dict] = []

    def add_record(self, record_type: str, content: str):
        self.records.append({"type": record_type, "content": content})

    def get_trajectory(self) -> str:
        # 格式化历史轨迹
        ...

    def get_last_execution(self) -> str:
        # 获取最近的执行结果
        ...
```

### 2. 流式输出支持

```python
def run_stream(self, input_text: str):
    # 初始执行
    yield from self._execute_phase(...)

    # 迭代优化
    for i in range(self.max_iterations):
        # 反思
        reflection = yield from self._reflection_phase(...)

        # 检查是否满意
        if self._is_satisfactory(reflection):
            break

        # 优化
        yield from self._execute_phase(...)
```

### 3. 结果解析

```python
def _parse_result(self, text: str) -> str:
    # 查找 Finish 标记
    finish_match = re.search(r"Finish\[(.*?)\]", text, re.DOTALL)
    if finish_match:
        return finish_match.group(1).strip()
    return text.strip()
```

---

## 十一、Reflection 的局限性

### ❌ 1. 计算成本高

每次迭代都需要完整的 LLM 调用，token 消耗大。

### ❌ 2. 反思质量依赖 LLM

如果 LLM 的反思能力不足，可能无法发现真正的问题。

### ❌ 3. 可能陷入局部最优

在某些情况下，可能会在某个局部解附近震荡。

---

## 十二、与其他技术的结合

| 结合技术            | 效果                              |
| ------------------ | --------------------------------- |
| Reflection + ReAct | 边行动边反思，在探索中持续改进       |
| Reflection + Tools | 使用工具验证反思结果的正确性         |
| Reflection + Memory | 长期记忆避免重复犯错                |

---

## 十三、未来扩展方向

1. **多智能体反思**：使用多个 Evaluator 进行交叉验证
2. **外部验证**：集成测试框架验证代码正确性
3. **学习式反思**：从历史反思中学习，改进反思策略
4. **分层反思**：先进行高层反思，再进行细节优化

---

## 十四、使用示例

```python
from ver_agent.agents import ReflectionAgent
from ver_agent.core import VerAgentLLM

# 创建 Reflection Agent
llm = VerAgentLLM()
agent = ReflectionAgent(
    name="Coder",
    llm=llm,
    max_iterations=3
)

# 执行任务
result = agent.run("编写一个 Python 快速排序函数")

# 查看反思过程
print(agent.memory.get_trajectory())
```

---

## 十五、参考文献

1. Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023)
2. ReAct: Synergizing Reasoning and Acting (Yao et al., 2022)
3. Self-Refine: Language Models with Human Feedback (Madaan et al., 2023)
4. CRITIC: Large Language Models Can Self-Correct with User Critiques (Wu et al., 2023)
