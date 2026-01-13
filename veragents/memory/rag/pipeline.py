"""RAG 管道占位实现。"""

from __future__ import annotations

from typing import List, Optional

from loguru import logger as log

from veragents.core.llm import VerAgentsLLM
from veragents.memory.base import BaseMemory
from veragents.memory.rag.document import Document


class RAGPipeline:
    """端到端的检索增强生成管道。"""

    def __init__(self, retriever: Optional[BaseMemory] = None, llm: Optional[VerAgentsLLM] = None):
        self.retriever = retriever
        self.llm = llm
        log.info(
            "RAGPipeline initialized | retriever={} llm={}",
            type(retriever).__name__ if retriever else None,
            getattr(llm, "provider", None) if llm else None,
        )

    def ingest(self, documents: List[Document]) -> None:
        """索引文档/知识库。"""
        log.warning("RAGPipeline.ingest not implemented | docs={}", len(documents))
        raise NotImplementedError("RAGPipeline.ingest not implemented yet")

    def query(self, question: str, top_k: int = 5, **kwargs):
        """执行检索与生成。"""
        log.warning("RAGPipeline.query not implemented | question_len={} top_k={}", len(question), top_k)
        raise NotImplementedError("RAGPipeline.query not implemented yet")


__all__ = ["RAGPipeline"]
