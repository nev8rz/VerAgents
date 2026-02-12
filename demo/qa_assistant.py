#!/usr/bin/env python
"""智能文档问答助手 Demo (8.4)

基于 VerAgents 的 RAG Pipeline 与 Memory Manager，构建交互式 PDF 学习助手。

功能：
1. 智能文档处理：MarkItDown 转换 → Markdown 智能分块 → 向量化索引
2. 高级检索问答：基础检索 / MQE 多查询扩展 / HyDE 假设文档嵌入
3. 多层次记忆管理：工作记忆、情景记忆、语义记忆协同
4. 个性化学习支持：笔记记录、学习回顾、统计报告

启动方式：
    python demo/qa_assistant.py

运行后访问 http://localhost:7860

前置条件：
    pip install gradio markitdown
    配置 .env：QDRANT_URL, QDRANT_API_KEY, EMBED_*, PROVIDER 等
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from loguru import logger as log

from veragents.memory.base import MemoryConfig
from veragents.memory.manager import MemoryManager
from veragents.memory.rag.pipeline import RAGPipeline

MAX_QA_SOURCES = 2


# ================================================================== #
#  辅助函数
# ================================================================== #


def _format_heading_for_display(heading: str, max_chars: int = 56) -> str:
    """清洗与截断 heading_path，避免展示 OCR 噪声。"""
    clean = (heading or "").strip()
    if not clean:
        return ""
    parts = [p.strip() for p in clean.split("/") if p.strip()]
    if parts:
        clean = parts[-1]
    clean = re.sub(r"^[#>\-\s]+", "", clean)
    clean = " ".join(clean.split())
    if not clean:
        return ""
    if len(clean) > max_chars:
        clean = clean[: max_chars - 3].rstrip() + "..."
    # 噪声标题直接不展示
    punct = sum(1 for ch in clean if not ch.isalnum() and not ("\u4e00" <= ch <= "\u9fff"))
    if punct / max(len(clean), 1) > 0.35:
        return ""
    return clean


def _preview_content(text: str, max_chars: int = 88) -> str:
    clean = " ".join((text or "").split())
    if len(clean) > max_chars:
        return clean[: max_chars - 3].rstrip() + "..."
    return clean


# ================================================================== #
#  核心助手类 PDFLearningAssistant
# ================================================================== #


class PDFLearningAssistant:
    """智能文档问答助手

    封装 RAGPipeline 和 MemoryManager 的调用逻辑，提供完整的文档问答工作流：
    - 文档载入与索引
    - 智能检索与问答
    - 学习笔记管理
    - 学习统计与报告
    """

    def __init__(self, user_id: str = "default_user"):
        """初始化学习助手

        Args:
            user_id: 用户 ID，用于隔离不同用户的数据
        """
        self.user_id = user_id
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.initialized = False

        # 延迟初始化的组件
        self._memory_manager: Optional[MemoryManager] = None
        self._rag_pipeline: Optional[RAGPipeline] = None

        # 学习统计
        self.stats = {
            "session_start": datetime.now(),
            "documents_loaded": 0,
            "questions_asked": 0,
            "concepts_learned": 0,
            "notes": [],
        }

        # 当前加载的文档
        self.current_document: Optional[str] = None

    def initialize(self) -> Dict[str, Any]:
        """初始化 RAG 和 Memory 子系统（连接数据库、加载模型等）

        Returns:
            Dict: 初始化结果
        """
        try:
            start = time.time()

            # 初始化 MemoryManager
            config = MemoryConfig(
                storage_path=f"./data/assistant_{self.user_id}",
                max_capacity=200,
            )
            self._memory_manager = MemoryManager(
                config=config,
                user_id=self.user_id,
                enable_working=True,
                enable_episodic=True,
                enable_semantic=True,
                enable_perceptual=False,
            )

            # 初始化 RAGPipeline
            self._rag_pipeline = RAGPipeline(
                knowledge_base_path=f"./data/kb_{self.user_id}",
                collection_name=f"qa_assistant_{self.user_id}",
                rag_namespace=f"pdf_{self.user_id}",
                chunk_tokens=512,
                overlap_tokens=64,
            )

            elapsed = time.time() - start
            self.initialized = True

            # 检查知识库是否已有数据（上次加载的）
            existing_msg = ""
            try:
                rag_stats = self._rag_pipeline.get_stats()
                vs = rag_stats.get("vector_store", {})
                points = vs.get("points_count", 0)
                if points > 0:
                    self.current_document = "(已有知识库)"
                    existing_msg = f"\n- 📚 检测到已有知识库: {points} 条向量，可直接提问！"
            except Exception:
                pass

            # 记录到情景记忆
            self._memory_manager.add_memory(
                content=f"学习助手初始化完成，会话 {self.session_id}",
                memory_type="episodic",
                importance=0.8,
                metadata={"event_type": "system_init", "session_id": self.session_id},
                auto_classify=False,
            )

            return {
                "success": True,
                "message": f"✅ 初始化成功！(耗时 {elapsed:.1f}s)\n"
                           f"- Memory: Working + Episodic + Semantic\n"
                           f"- RAG: Qdrant + SQLite + Embedding({self._rag_pipeline.dimension}d)\n"
                           f"- 会话: {self.session_id}"
                           f"{existing_msg}",
            }
        except Exception as e:
            log.exception("初始化失败")
            return {"success": False, "message": f"❌ 初始化失败: {e}"}

    # ================================================================== #
    #  文档处理
    # ================================================================== #

    def load_document(self, pdf_path: str) -> Dict[str, Any]:
        """加载 PDF 文档到知识库

        流程：MarkItDown 转换 → 智能分块 → 向量化 → Qdrant + SQLite

        Args:
            pdf_path: PDF 文件路径

        Returns:
            Dict: 包含 success 和 message 的结果
        """
        if not self.initialized:
            return {"success": False, "message": "⚠️ 请先初始化助手！"}

        if not pdf_path or not os.path.exists(pdf_path):
            return {"success": False, "message": f"⚠️ 文件不存在: {pdf_path}"}

        start_time = time.time()

        try:
            # 【RAGPipeline】处理文档
            chunk_count = self._rag_pipeline.ingest_file(pdf_path)
            process_time = time.time() - start_time

            if chunk_count > 0:
                self.current_document = os.path.basename(pdf_path)
                self.stats["documents_loaded"] += 1

                # 【MemoryManager】记录到情景记忆
                self._memory_manager.add_memory(
                    content=f"加载了文档《{self.current_document}》，生成 {chunk_count} 个分块",
                    memory_type="episodic",
                    importance=0.9,
                    metadata={
                        "event_type": "document_loaded",
                        "session_id": self.session_id,
                        "file": pdf_path,
                        "chunks": chunk_count,
                    },
                    auto_classify=False,
                )

                return {
                    "success": True,
                    "message": (
                        f"✅ 文档载入成功！\n"
                        f"- 文件: {self.current_document}\n"
                        f"- 分块数: {chunk_count}\n"
                        f"- 耗时: {process_time:.1f}s"
                    ),
                }
            else:
                return {"success": False, "message": "⚠️ 文档解析后无有效内容"}

        except Exception as e:
            log.exception("文档载入失败")
            return {"success": False, "message": f"❌ 加载失败: {e}"}

    # ================================================================== #
    #  智能问答
    # ================================================================== #

    def ask(self, question: str, use_mqe: bool = False, use_hyde: bool = False) -> str:
        """向文档提问

        Args:
            question: 用户问题
            use_mqe: 启用多查询扩展
            use_hyde: 启用假设文档嵌入

        Returns:
            str: 格式化的回答
        """
        if not self.initialized:
            return "⚠️ 请先初始化助手！"
        if not self.current_document:
            return "⚠️ 请先加载文档！"
        if not question or not question.strip():
            return "⚠️ 请输入问题！"

        # 【MemoryManager】记录问题到工作记忆
        self._memory_manager.add_memory(
            content=f"用户提问: {question}",
            memory_type="working",
            importance=0.6,
            metadata={"session_id": self.session_id, "event_type": "question"},
            auto_classify=False,
        )

        start_time = time.time()

        # 【RAGPipeline】检索 + LLM 问答
        try:
            result = self._rag_pipeline.query(
                question=question,
                top_k=MAX_QA_SOURCES,
                enable_mqe=use_mqe,
                enable_hyde=use_hyde,
            )
        except Exception as e:
            log.exception("RAG 问答失败")
            return f"❌ 问答出错: {e}"

        elapsed = time.time() - start_time

        # 【MemoryManager】记录到情景记忆
        self._memory_manager.add_memory(
            content=f"关于「{question}」的问答 — 检索到 {len(result['sources'])} 条参考",
            memory_type="episodic",
            importance=0.7,
            metadata={
                "event_type": "qa_interaction",
                "session_id": self.session_id,
                "question": question,
            },
            auto_classify=False,
        )

        self.stats["questions_asked"] += 1

        # 格式化输出
        answer = result.get("answer", "未找到相关信息")
        sources = result.get("sources", [])

        output_parts = [f"💡 **回答**\n\n{answer}\n"]

        if sources:
            output_parts.append("\n📚 **参考来源**\n")
            for i, src in enumerate(sources[:MAX_QA_SOURCES], 1):
                score = float(src.get("score", 0.0) or 0.0)
                rank_score = float(src.get("rank_score", score) or score)
                heading = _format_heading_for_display(src.get("heading_path", ""))
                source_name = os.path.basename(src.get("source", "")) or "直接输入"
                heading_str = f" [{heading}]" if heading else ""
                preview = _preview_content(src.get("content", ""))
                output_parts.append(
                    f"{i}. `rank={rank_score:.3f}` (vec={score:.3f}) {source_name}{heading_str}\n"
                    f"   {preview}"
                )

        strategy = []
        if use_mqe:
            strategy.append("MQE")
        if use_hyde:
            strategy.append("HyDE")
        strategy_str = " + ".join(strategy) if strategy else "基础检索"
        output_parts.append(f"\n⏱️ 耗时 {elapsed:.1f}s | 策略: {strategy_str}")

        return "\n".join(output_parts)

    # ================================================================== #
    #  学习笔记
    # ================================================================== #

    def add_note(self, content: str, concept: str = "") -> str:
        """添加学习笔记到语义记忆

        Args:
            content: 笔记内容
            concept: 关联的概念/主题

        Returns:
            str: 操作结果
        """
        if not self.initialized:
            return "⚠️ 请先初始化助手！"
        if not content or not content.strip():
            return "⚠️ 请输入笔记内容！"

        concept = concept.strip() or "通用笔记"

        # 【MemoryManager】存储到语义记忆
        memory_id = self._memory_manager.add_memory(
            content=f"[{concept}] {content}",
            memory_type="semantic",
            importance=0.8,
            metadata={
                "concept": concept,
                "session_id": self.session_id,
                "event_type": "note",
                "document": self.current_document or "",
            },
            auto_classify=False,
        )

        self.stats["concepts_learned"] += 1
        self.stats["notes"].append({
            "concept": concept,
            "content": content,
            "time": datetime.now().strftime("%H:%M:%S"),
        })

        return f"✅ 笔记已保存！\n- 概念: {concept}\n- ID: {memory_id[:8]}...\n- 累计笔记: {self.stats['concepts_learned']} 条"

    # ================================================================== #
    #  学习回顾
    # ================================================================== #

    def recall(self, query: str, limit: int = 5) -> str:
        """回顾学习历程 — 从记忆系统中检索

        Args:
            query: 检索关键词
            limit: 返回数量

        Returns:
            str: 格式化的记忆检索结果
        """
        if not self.initialized:
            return "⚠️ 请先初始化助手！"
        if not query or not query.strip():
            return "⚠️ 请输入回顾关键词！"

        results = self._memory_manager.retrieve_memories(query=query, limit=limit)

        if not results:
            return f"🔍 没有找到与「{query}」相关的学习记忆。"

        output_parts = [f"🔍 **回顾「{query}」** — 找到 {len(results)} 条记忆\n"]
        for i, item in enumerate(results, 1):
            mtype = getattr(item, "memory_type", "unknown")
            importance = getattr(item, "importance", 0)
            ts = getattr(item, "timestamp", None)
            ts_str = ts.strftime("%m-%d %H:%M") if ts else ""
            content = item.content[:80].replace("\n", " ")
            output_parts.append(
                f"{i}. [{mtype}] (重要性={importance:.1f}) {ts_str}\n   {content}..."
            )

        return "\n".join(output_parts)

    # ================================================================== #
    #  统计与报告
    # ================================================================== #

    def get_stats_text(self) -> str:
        """获取学习统计（文本格式）"""
        if not self.initialized:
            return "⚠️ 请先初始化助手！"

        duration = (datetime.now() - self.stats["session_start"]).total_seconds()
        minutes = int(duration // 60)
        seconds = int(duration % 60)

        # Memory 统计
        mem_stats = self._memory_manager.get_memory_stats()

        # RAG 统计
        rag_stats = self._rag_pipeline.get_stats()

        lines = [
            "📊 **学习统计**\n",
            f"⏱️ 会话时长: {minutes} 分 {seconds} 秒",
            f"📄 已加载文档: {self.stats['documents_loaded']}",
            f"❓ 提问次数: {self.stats['questions_asked']}",
            f"📝 学习笔记: {self.stats['concepts_learned']}",
            f"📖 当前文档: {self.current_document or '未加载'}",
            "",
            "**记忆系统**",
            f"- 总记忆数: {mem_stats.get('total_memories', 0)}",
        ]

        for mtype, mstat in mem_stats.get("memories_by_type", {}).items():
            count = mstat.get("count", 0)
            lines.append(f"- {mtype}: {count} 条")

        vs = rag_stats.get("vector_store", {})
        lines.extend([
            "",
            "**RAG 知识库**",
            f"- 向量点数: {vs.get('points_count', 0)}",
            f"- 向量维度: {rag_stats.get('dimension', '?')}",
            f"- 命名空间: {rag_stats.get('namespace', '?')}",
        ])

        return "\n".join(lines)

    def generate_report(self) -> str:
        """生成学习报告（JSON 格式）"""
        if not self.initialized:
            return "⚠️ 请先初始化助手！"

        duration = (datetime.now() - self.stats["session_start"]).total_seconds()
        mem_stats = self._memory_manager.get_memory_stats()
        rag_stats = self._rag_pipeline.get_stats()

        report = {
            "session_info": {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "start_time": self.stats["session_start"].isoformat(),
                "duration_seconds": round(duration, 1),
            },
            "learning_metrics": {
                "documents_loaded": self.stats["documents_loaded"],
                "questions_asked": self.stats["questions_asked"],
                "concepts_learned": self.stats["concepts_learned"],
                "current_document": self.current_document,
            },
            "notes": self.stats["notes"],
            "memory_summary": {
                "total": mem_stats.get("total_memories", 0),
                "by_type": {
                    k: v.get("count", 0)
                    for k, v in mem_stats.get("memories_by_type", {}).items()
                },
            },
            "rag_status": {
                "namespace": rag_stats.get("namespace"),
                "total_chunks": rag_stats.get("total_chunks"),
                "dimension": rag_stats.get("dimension"),
                "vector_points": rag_stats.get("vector_store", {}).get("points_count", 0),
            },
        }

        # 保存到文件
        os.makedirs("./data/reports", exist_ok=True)
        report_file = f"./data/reports/report_{self.session_id}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        return json.dumps(report, ensure_ascii=False, indent=2, default=str)


# ================================================================== #
#  Gradio Web 界面
# ================================================================== #


def build_gradio_app():
    """构建 Gradio Web 界面"""
    import gradio as gr

    assistant: Optional[PDFLearningAssistant] = None

    # ---- 回调函数 ----

    def on_init(user_id: str):
        nonlocal assistant
        uid = user_id.strip() or "default_user"
        assistant = PDFLearningAssistant(user_id=uid)
        result = assistant.initialize()
        return result["message"]

    def on_load_doc(file):
        if assistant is None or not assistant.initialized:
            return "⚠️ 请先初始化助手！"
        if file is None:
            return "⚠️ 请上传文件！"
        # Gradio 上传的文件路径
        path = file.name if hasattr(file, "name") else str(file)
        result = assistant.load_document(path)
        return result["message"]

    def on_ask(question: str, use_mqe: bool, use_hyde: bool):
        if assistant is None or not assistant.initialized:
            return "⚠️ 请先初始化助手！"
        return assistant.ask(question, use_mqe=use_mqe, use_hyde=use_hyde)

    def on_add_note(content: str, concept: str):
        if assistant is None or not assistant.initialized:
            return "⚠️ 请先初始化助手！"
        return assistant.add_note(content, concept)

    def on_recall(query: str):
        if assistant is None or not assistant.initialized:
            return "⚠️ 请先初始化助手！"
        return assistant.recall(query)

    def on_stats():
        if assistant is None or not assistant.initialized:
            return "⚠️ 请先初始化助手！"
        return assistant.get_stats_text()

    def on_report():
        if assistant is None or not assistant.initialized:
            return "⚠️ 请先初始化助手！"
        return assistant.generate_report()

    # ---- 构建界面 ----

    with gr.Blocks(
        title="VerAgents 智能文档问答助手",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
        ),
        css="""
        .main-title { text-align: center; margin-bottom: 0.5em; }
        .sub-title { text-align: center; color: #666; margin-bottom: 1.5em; }
        """,
    ) as app:
        gr.Markdown(
            "# 📚 VerAgents 智能文档问答助手",
            elem_classes="main-title",
        )
        gr.Markdown(
            "基于 RAG Pipeline + Memory Manager，实现文档智能问答、学习笔记与进度追踪",
            elem_classes="sub-title",
        )

        # ---- Tab 0: 初始化与文档加载 ----
        with gr.Tab("🏠 文档管理"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1️⃣ 初始化助手")
                    user_id_input = gr.Textbox(
                        label="用户 ID",
                        value="default_user",
                        placeholder="输入用户 ID（不同用户数据隔离）",
                    )
                    init_btn = gr.Button("🚀 初始化", variant="primary")
                    init_output = gr.Textbox(label="初始化结果", lines=5, interactive=False)

                with gr.Column(scale=1):
                    gr.Markdown("### 2️⃣ 加载文档")
                    file_input = gr.File(
                        label="上传 PDF / Word / Markdown 文件",
                        file_types=[".pdf", ".docx", ".md", ".txt", ".html", ".csv", ".json"],
                    )
                    load_btn = gr.Button("📄 加载文档", variant="primary")
                    load_output = gr.Textbox(label="加载结果", lines=5, interactive=False)

            init_btn.click(fn=on_init, inputs=[user_id_input], outputs=[init_output])
            load_btn.click(fn=on_load_doc, inputs=[file_input], outputs=[load_output])

        # ---- Tab 1: 智能问答 ----
        with gr.Tab("❓ 智能问答"):
            gr.Markdown("### 向已加载的文档提问")

            question_input = gr.Textbox(
                label="你的问题",
                placeholder="例如：什么是 Transformer？它的核心机制是什么？",
                lines=2,
            )

            with gr.Row():
                mqe_checkbox = gr.Checkbox(label="🔀 多查询扩展 (MQE)", value=False)
                hyde_checkbox = gr.Checkbox(label="🧠 假设文档嵌入 (HyDE)", value=False)

            ask_btn = gr.Button("🔍 提问", variant="primary")
            answer_output = gr.Markdown(label="回答")

            ask_btn.click(
                fn=on_ask,
                inputs=[question_input, mqe_checkbox, hyde_checkbox],
                outputs=[answer_output],
            )

        # ---- Tab 2: 学习笔记 ----
        with gr.Tab("📝 学习笔记"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 添加笔记")
                    concept_input = gr.Textbox(
                        label="概念/主题",
                        placeholder="例如：注意力机制",
                    )
                    note_content = gr.Textbox(
                        label="笔记内容",
                        placeholder="写下你的理解或总结...",
                        lines=4,
                    )
                    note_btn = gr.Button("💾 保存笔记", variant="primary")
                    note_output = gr.Textbox(label="保存结果", lines=3, interactive=False)

                with gr.Column(scale=1):
                    gr.Markdown("### 回顾记忆")
                    recall_input = gr.Textbox(
                        label="回顾关键词",
                        placeholder="输入关键词搜索学习记忆...",
                    )
                    recall_btn = gr.Button("🔍 回顾", variant="secondary")
                    recall_output = gr.Markdown(label="记忆检索结果")

            note_btn.click(fn=on_add_note, inputs=[note_content, concept_input], outputs=[note_output])
            recall_btn.click(fn=on_recall, inputs=[recall_input], outputs=[recall_output])

        # ---- Tab 3: 学习统计 ----
        with gr.Tab("📊 学习统计"):
            with gr.Row():
                stats_btn = gr.Button("📊 查看统计", variant="secondary")
                report_btn = gr.Button("📋 生成报告", variant="primary")

            stats_output = gr.Markdown(label="统计信息")
            report_output = gr.Code(label="学习报告 (JSON)", language="json")

            stats_btn.click(fn=on_stats, outputs=[stats_output])
            report_btn.click(fn=on_report, outputs=[report_output])

    return app


# ================================================================== #
#  主入口
# ================================================================== #


def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║    VerAgents 智能文档问答助手 — Gradio Web Demo         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    app = build_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
