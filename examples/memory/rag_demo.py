#!/usr/bin/env python
"""RAG 系统完整使用示例。

演示功能：
1. 初始化 RAG Pipeline
2. 文本载入与智能分块
3. 文件载入（MarkItDown 转换）
4. 基础向量检索
5. 高级检索策略（MQE / HyDE）
6. LLM 增强问答
7. 知识库管理与统计
8. 清空知识库

运行前请确保设置好环境变量：
- QDRANT_URL / QDRANT_API_KEY
- EMBED_BASE_URL / EMBED_API_KEY / EMBED_MODEL_NAME
- PROVIDER, ZHIPU_BASE_URL, ZHIPU_API_KEY 等（LLM 问答功能需要）
"""

from __future__ import annotations

import os
import sys
import tempfile

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

load_dotenv()


def separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def demo_1_init_pipeline():
    """1. 初始化 RAG Pipeline"""
    separator("1. 初始化 RAG Pipeline")

    from veragents.memory.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline(
        knowledge_base_path="./knowledge_base_demo",
        collection_name="rag_demo",
        rag_namespace="demo",
        chunk_tokens=256,
        overlap_tokens=32,
    )

    print(f"✅ Pipeline 初始化成功")
    print(f"   知识库路径: {pipeline.knowledge_base_path}")
    print(f"   向量集合: {pipeline.collection_name}")
    print(f"   命名空间: {pipeline.rag_namespace}")
    print(f"   嵌入维度: {pipeline.dimension}")
    print(f"   分块大小: {pipeline.chunk_tokens} tokens")
    print(f"   重叠: {pipeline.overlap_tokens} tokens")

    return pipeline


def demo_2_smart_chunking():
    """2. 智能 Markdown 分块演示"""
    separator("2. 智能 Markdown 分块")

    from veragents.memory.rag.pipeline import smart_chunk_markdown, _approx_token_len

    sample_md = """# 深度学习基础

## 1. 神经网络概述

神经网络是一种模拟人脑神经元连接的计算模型。它由输入层、隐藏层和输出层组成。
每个神经元接收输入信号，经过加权求和和激活函数处理后，产生输出。

深度学习是使用多层神经网络的机器学习方法。随着层数的增加，网络能够学习到越来越抽象的特征表示。

## 2. 常见架构

### 2.1 卷积神经网络 (CNN)

CNN 专门用于处理具有网格结构的数据，如图像。其核心操作包括：
- **卷积层**：使用卷积核提取局部特征
- **池化层**：降低特征图的空间维度
- **全连接层**：进行最终的分类或回归

### 2.2 循环神经网络 (RNN)

RNN 擅长处理序列数据，如文本和时间序列。它通过隐藏状态在时间步之间传递信息。
LSTM 和 GRU 是 RNN 的改进版本，解决了长距离依赖问题。

### 2.3 Transformer

Transformer 基于自注意力机制，完全摒弃了循环结构。它是 BERT、GPT 等大语言模型的基础。
自注意力机制允许模型在处理每个位置时，同时关注输入序列的所有位置。

## 3. 训练技术

### 3.1 优化器

- Adam：自适应学习率优化器，结合了 Momentum 和 RMSProp 的优点
- SGD：随机梯度下降，是最基础的优化算法
- AdamW：在 Adam 基础上加入权重衰减

### 3.2 正则化

- Dropout：随机丢弃神经元，防止过拟合
- Batch Normalization：标准化层输入，加速训练
- Label Smoothing：软化标签，提高泛化能力
"""

    chunks = smart_chunk_markdown(sample_md, chunk_tokens=100, overlap_tokens=16)

    print(f"原始文本长度: {len(sample_md)} chars, ~{_approx_token_len(sample_md)} tokens")
    print(f"分块数量: {len(chunks)}")
    print()

    for i, chunk in enumerate(chunks):
        print(f"--- 分块 {i} ({chunk['token_estimate']} tokens) ---")
        heading = chunk.get("heading_path", "")
        if heading:
            print(f"    标题路径: {heading}")
        content_preview = chunk["content"][:80].replace("\n", " ")
        print(f"    内容: {content_preview}...")
        print()

    return sample_md


def demo_3_ingest_text(pipeline):
    """3. 文本载入"""
    separator("3. 载入文本到知识库")

    knowledge_texts = [
        (
            "Python 编程语言",
            """# Python 编程语言

Python 是一种高级、通用的编程语言。它的设计哲学强调代码的可读性和简洁性。

## 核心特性

- **动态类型**：变量不需要声明类型
- **解释执行**：代码逐行解释执行，无需编译
- **丰富的标准库**：内置大量实用模块
- **多范式支持**：支持面向对象、函数式等多种编程范式

## 常见应用

Python 广泛应用于 Web 开发、数据科学、人工智能、自动化运维等领域。
Django 和 Flask 是流行的 Web 框架。
NumPy、Pandas 和 Scikit-learn 是数据科学的核心库。
TensorFlow 和 PyTorch 是主流的深度学习框架。
""",
        ),
        (
            "向量数据库介绍",
            """# 向量数据库

向量数据库是专门用于存储和检索高维向量的数据库系统。

## 工作原理

向量数据库通过近似最近邻（ANN）算法来实现高效的相似性搜索。
常用的 ANN 算法包括 HNSW、IVF、PQ 等。

## 主流产品

- **Qdrant**：基于 Rust 开发，支持过滤检索
- **Milvus**：CNCF 项目，支持多种索引类型
- **Pinecone**：全托管云服务
- **Weaviate**：支持模块化向量化

## 应用场景

1. RAG（检索增强生成）
2. 推荐系统
3. 图像/音频搜索
4. 异常检测
""",
        ),
        (
            "大语言模型概述",
            """# 大语言模型 (LLM)

大语言模型是基于 Transformer 架构的超大规模语言模型。

## 代表模型

- **GPT 系列**：OpenAI 的自回归语言模型
- **BERT**：Google 的双向编码模型
- **LLaMA**：Meta 开源的大模型系列
- **通义千问**：阿里的大模型
- **GLM**：智谱的对话大模型

## RAG 技术

RAG（Retrieval-Augmented Generation）是将检索和生成结合的技术。
核心思想是在生成回答前，先从知识库中检索相关文档。
这样可以减少大模型的幻觉问题，提高回答的准确性和时效性。

## 微调技术

- LoRA：低秩适应，参数高效微调
- QLoRA：量化低秩适应
- Full Fine-tuning：全量微调
- Instruction Tuning：指令微调
""",
        ),
    ]

    total_chunks = 0
    for source_name, text in knowledge_texts:
        count = pipeline.ingest_text(text, source=source_name)
        total_chunks += count
        print(f"✅ 已载入: {source_name} → {count} 个分块")

    print(f"\n总计载入: {total_chunks} 个分块")
    return total_chunks


def demo_4_ingest_file(pipeline):
    """4. 文件载入（MarkItDown 转换）"""
    separator("4. 文件载入")

    # 创建临时测试文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write("""# VerAgents 框架介绍

VerAgents 是一个模块化的 AI Agent 框架。

## 核心模块

### 记忆系统

VerAgents 的记忆系统包含四种记忆类型：
- **工作记忆**：短期上下文，容量有限
- **情景记忆**：具体交互事件记录
- **语义记忆**：概念和知识存储
- **感知记忆**：多模态数据处理

### RAG 系统

RAG 系统是一种工具，提供文档载入、智能检索和 LLM 增强问答功能。
支持多种文档格式，包括 PDF、Word、Excel 等。

### 工具系统

VerAgents 提供可扩展的工具注册机制，支持自定义工具。
""")
        tmp_path = f.name

    try:
        count = pipeline.ingest_file(tmp_path)
        print(f"✅ 文件载入成功: {os.path.basename(tmp_path)} → {count} 个分块")
    finally:
        os.unlink(tmp_path)

    return count


def demo_5_basic_search(pipeline):
    """5. 基础向量检索"""
    separator("5. 基础向量检索")

    queries = [
        "Python 可以用来做什么",
        "什么是向量数据库",
        "RAG 技术的原理",
        "深度学习框架有哪些",
    ]

    for query in queries:
        print(f"🔍 查询: {query}")
        results = pipeline.search(query=query, top_k=3)
        for i, r in enumerate(results, 1):
            score = r["score"]
            heading = r.get("heading_path", "")
            heading_str = f" [{heading}]" if heading else ""
            content_preview = r["content"][:60].replace("\n", " ")
            print(f"   {i}. [score={score:.3f}]{heading_str} {content_preview}...")
        print()


def demo_6_advanced_search(pipeline):
    """6. 高级检索策略"""
    separator("6. 高级检索策略（MQE / HyDE）")

    query = "如何使用 RAG 技术提高问答准确性"

    # 基础检索
    print(f"🔍 原始查询: {query}\n")

    print("--- 基础检索 ---")
    results_basic = pipeline.search(query=query, top_k=3)
    for i, r in enumerate(results_basic, 1):
        content_preview = r["content"][:60].replace("\n", " ")
        print(f"   {i}. [score={r['score']:.3f}] {content_preview}...")

    # MQE 检索
    print("\n--- MQE 多查询扩展检索 ---")
    try:
        results_mqe = pipeline.search(query=query, top_k=3, enable_mqe=True, mqe_expansions=2)
        for i, r in enumerate(results_mqe, 1):
            content_preview = r["content"][:60].replace("\n", " ")
            print(f"   {i}. [score={r['score']:.3f}] {content_preview}...")
    except Exception as e:
        print(f"   ⚠️ MQE 需要 LLM 支持: {e}")

    # HyDE 检索
    print("\n--- HyDE 假设文档检索 ---")
    try:
        results_hyde = pipeline.search(query=query, top_k=3, enable_hyde=True)
        for i, r in enumerate(results_hyde, 1):
            content_preview = r["content"][:60].replace("\n", " ")
            print(f"   {i}. [score={r['score']:.3f}] {content_preview}...")
    except Exception as e:
        print(f"   ⚠️ HyDE 需要 LLM 支持: {e}")


def demo_7_rag_query(pipeline):
    """7. LLM 增强问答"""
    separator("7. LLM 增强问答")

    questions = [
        "Python 有哪些主要的应用领域？",
        "RAG 技术是什么？它能解决什么问题？",
    ]

    for question in questions:
        print(f"❓ 问题: {question}\n")
        try:
            result = pipeline.query(question=question, top_k=3)
            print(f"💡 回答:\n{result['answer']}\n")
            print(f"📚 参考来源: {result['sources_count'] if 'sources_count' in result else len(result['sources'])} 条")
            for i, src in enumerate(result["sources"][:3], 1):
                source_name = src.get("source", "unknown")
                print(f"   {i}. [{src['score']:.3f}] {source_name}")
        except Exception as e:
            print(f"   ⚠️ LLM 问答需要配置 Provider: {e}")
        print()


def demo_8_stats(pipeline):
    """8. 知识库统计"""
    separator("8. 知识库统计")

    stats = pipeline.get_stats()

    print(f"命名空间: {stats['namespace']}")
    print(f"向量集合: {stats['collection']}")
    print(f"已索引文件数: {stats['indexed_files']}")
    print(f"总分块数: {stats['total_chunks']}")
    print(f"嵌入维度: {stats['dimension']}")
    print(f"分块配置: {stats['chunk_config']}")

    vs = stats.get("vector_store", {})
    if vs:
        print(f"向量存储: {vs}")

    ds = stats.get("document_store", {})
    if ds:
        print(f"文档存储: {ds}")


def demo_9_cleanup(pipeline):
    """9. 清空知识库"""
    separator("9. 清空知识库")

    pipeline.clear()
    print("✅ 知识库已清空")

    import shutil
    kb_path = "./knowledge_base_demo"
    if os.path.exists(kb_path):
        shutil.rmtree(kb_path)
        print(f"✅ 已删除临时目录: {kb_path}")


def main():
    """RAG 系统完整演示。"""
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         VerAgents RAG 系统 完整使用示例                 ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 1. 初始化
    pipeline = demo_1_init_pipeline()

    # 2. 分块演示（独立，不影响 pipeline）
    demo_2_smart_chunking()

    # 3. 载入文本
    demo_3_ingest_text(pipeline)

    # 4. 载入文件
    demo_4_ingest_file(pipeline)

    # 5. 基础检索
    demo_5_basic_search(pipeline)

    # 6. 高级检索（MQE/HyDE 需要 LLM）
    demo_6_advanced_search(pipeline)

    # 7. LLM 问答
    demo_7_rag_query(pipeline)

    # 8. 统计
    demo_8_stats(pipeline)

    # 9. 清理
    demo_9_cleanup(pipeline)

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          ✅ RAG 系统演示完成                            ║")
    print("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
