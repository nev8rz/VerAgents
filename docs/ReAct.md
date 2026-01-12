# ReAct
---

## 一、论文基本信息

**论文标题**

> *ReAct: Synergizing Reasoning and Acting in Language Models*

**作者**
Shunyu Yao et al.（Google / Princeton）

**首次提出时间**
2022 年

**核心关键词**

* Reasoning（推理）
* Acting（行动）
* Tool Use（工具调用）
* Chain-of-Thought（思维链）

---

## 二、研究动机（为什么要 ReAct）

### 1️⃣ 之前方法的两个极端问题

### （1）只有 Reasoning（CoT）

如 Chain-of-Thought：

```text
Thought: Let's think step by step...
```

**问题**：

* 模型只能“想”，不能“查”
* 容易 hallucination（编造事实）
* 无法与环境交互

---

### （2）只有 Acting（Tool-Only）

如早期 tool calling：

```json
{
  "action": "search",
  "input": "人口最多的国家"
}
```

**问题**：

* 没有显式推理过程
* 行为不稳定
* 难以调试和泛化

---

### ✅ ReAct 的核心动机

> **把“推理过程”和“行动过程”交织在一起，让模型边想边做。**

---

## 三、ReAct 的核心思想（一句话）

> **让语言模型在一个循环中交替生成：**
>
> **Thought → Action → Observation → Thought → … → Answer**

这是 ReAct 的灵魂。

---

## 四、ReAct 的整体框架

### ReAct 交互格式（标准）

```text
Thought: ...
Action: ...
Observation: ...
Thought: ...
Action: ...
Observation: ...
...
Final Answer: ...
```

---

### 关键组成部分

| 组件          | 作用        |
| ----------- | --------- |
| Thought     | 模型的中间推理   |
| Action      | 调用工具 / 环境 |
| Observation | 工具返回的结果   |
| Answer      | 最终输出      |

---

## 五、ReAct 的工作流程（详细）

### 1️⃣ 用户提问

```text
Q: 爱因斯坦出生在哪个国家？
```

---

### 2️⃣ Thought（思考）

```text
Thought: 我需要知道爱因斯坦的出生地，可以查一下维基百科。
```

---

### 3️⃣ Action（执行）

```text
Action: Search["Albert Einstein birthplace"]
```

---

### 4️⃣ Observation（环境反馈）

```text
Observation: Albert Einstein was born in Ulm, Kingdom of Württemberg, German Empire.
```

---

### 5️⃣ Thought（再思考）

```text
Thought: 乌尔姆属于当时的德意志帝国，因此国家是德国。
```

---

### 6️⃣ Final Answer（最终答案）

```text
Final Answer: 爱因斯坦出生在德国。
```

---

## 六、ReAct 与 Chain-of-Thought 的本质区别

| 对比点        | CoT | ReAct |
| ---------- | --- | ----- |
| 是否调用工具     | ❌   | ✅     |
| 是否与环境交互    | ❌   | ✅     |
| 推理是否可纠错    | ❌   | ✅     |
| 是否减少幻觉     | 一般  | 显著    |
| 是否适合 Agent | ❌   | ✅     |

---

## 七、论文中的关键实验

### 1️⃣ 实验任务

* HotpotQA（多跳问答）
* FEVER（事实验证）
* ALFWorld（文本环境）
* Web-based QA

---

### 2️⃣ 实验对比方法

* Standard Prompt
* Chain-of-Thought
* Tool-only
* **ReAct**

---

### 3️⃣ 关键结论

> **ReAct 在需要外部知识或多步决策的任务上显著优于 CoT。**

特别是：

* 减少 hallucination
* 更强的泛化能力
* 更可解释

---

## 八、ReAct 的 Prompt 示例（论文原始风格）

```text
You are a reasoning agent.

Question: What is the capital of the country where the Eiffel Tower is located?

Thought: The Eiffel Tower is in Paris. Paris is in France. I should verify the capital of France.

Action: Search["capital of France"]

Observation: The capital of France is Paris.

Thought: The capital is Paris.

Final Answer: Paris.
```

---

## 九、ReAct 的优势总结

### ✅ 1. 可解释性极强

* 每一步 Thought 都可读
* 易于 Debug

### ✅ 2. 显著减少幻觉

* 不确定就查
* 查完再答

### ✅ 3. 自然适配 Agent 系统

* LangChain
* AutoGPT
* OpenAI function calling

---

## 十、ReAct 的局限性（论文也坦诚）

### ❌ 1. Prompt 复杂

* token 成本高
* 速度慢

### ❌ 2. 对工具依赖强

* 工具设计不好 → Agent 不稳定

### ❌ 3. 推理仍可能出错

* LLM 并非符号推理器

---

## 十一、ReAct 对今天 Agent 框架的影响

| 框架           | 关系               |
| ------------ | ---------------- |
| LangChain    | ReAct 是核心模式      |
| AutoGPT      | ReAct + Planning |
| OpenAI Tools | ReAct 的工业化       |
| BabyAGI      | ReAct + 记忆       |

---

## 十二、ReAct 的演化方向（重要）

* **ReAct + Planning**（Plan-and-Execute）
* **ReAct + Memory**
* **ReAct + Reflection**
* **Tree-of-Thoughts**

---

