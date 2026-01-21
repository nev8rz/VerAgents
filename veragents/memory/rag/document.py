"""RAG 文档处理模块。

提供：
- Document: 文档封装类
- DocumentChunk: 文档块封装类
- DocumentProcessor: 文档分块处理器
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger as log

from veragents.memory.base import MemoryItem


@dataclass
class Document:
    """文档类。"""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
    source: Optional[str] = None

    def __post_init__(self):
        if self.doc_id is None:
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()
        if self.source:
            self.metadata["source"] = self.source

    @property
    def id(self) -> str:
        """兼容旧接口。"""
        return self.doc_id or ""

    def to_memory_item(self, user_id: str = "default", memory_type: str = "document") -> MemoryItem:
        """转换为通用 MemoryItem。"""
        return MemoryItem(
            id=self.doc_id or hashlib.md5(self.content.encode()).hexdigest(),
            content=self.content,
            memory_type=memory_type,
            user_id=user_id,
            timestamp=datetime.now(),
            metadata=self.metadata,
        )


@dataclass
class DocumentChunk:
    """文档块类。"""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    chunk_index: int = 0

    def __post_init__(self):
        if self.chunk_id is None:
            chunk_content = f"{self.doc_id}_{self.chunk_index}_{self.content[:50]}"
            self.chunk_id = hashlib.md5(chunk_content.encode()).hexdigest()

    @property
    def id(self) -> str:
        """兼容旧接口。"""
        return self.chunk_id or ""

    def to_memory_item(self, user_id: str = "default", memory_type: str = "document_chunk") -> MemoryItem:
        """转换为通用 MemoryItem。"""
        return MemoryItem(
            id=self.chunk_id or hashlib.md5(self.content.encode()).hexdigest(),
            content=self.content,
            memory_type=memory_type,
            user_id=user_id,
            timestamp=datetime.now(),
            metadata={
                **self.metadata,
                "doc_id": self.doc_id,
                "chunk_index": self.chunk_index,
            },
        )


class DocumentProcessor:
    """文档处理器。

    功能：
    - 文档分块
    - 块合并
    - 块过滤
    - 元数据管理
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", ".", " "]

        log.info(
            "DocumentProcessor initialized | chunk_size={} overlap={} separators={}",
            chunk_size,
            chunk_overlap,
            len(self.separators),
        )

    def process_document(self, document: Document) -> List[DocumentChunk]:
        """处理文档，分割成块。"""
        chunks = self._split_text(document.content)

        document_chunks = []
        for i, chunk_content in enumerate(chunks):
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update(
                {
                    "doc_id": document.doc_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "processed_at": datetime.now().isoformat(),
                }
            )

            chunk = DocumentChunk(
                content=chunk_content,
                metadata=chunk_metadata,
                doc_id=document.doc_id,
                chunk_index=i,
            )
            document_chunks.append(chunk)

        log.info(
            "DocumentProcessor processed | doc_id={}... chunks={}",
            (document.doc_id or "")[:8],
            len(document_chunks),
        )
        return document_chunks

    def process_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """批量处理文档。"""
        all_chunks = []
        for document in documents:
            chunks = self.process_document(document)
            all_chunks.extend(chunks)

        log.info("DocumentProcessor batch processed | docs={} total_chunks={}", len(documents), len(all_chunks))
        return all_chunks

    def _split_text(self, text: str) -> List[str]:
        """分割文本为块。"""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                chunks.append(text[start:])
                break

            split_point = self._find_split_point(text, start, end)

            if split_point == -1:
                split_point = end

            chunks.append(text[start:split_point])
            start = max(start + 1, split_point - self.chunk_overlap)

        return chunks

    def _find_split_point(self, text: str, start: int, end: int) -> int:
        """在指定范围内寻找最佳分割点。"""
        for separator in self.separators:
            search_start = max(start, end - 100)

            for i in range(end - len(separator), search_start - 1, -1):
                if text[i : i + len(separator)] == separator:
                    return i + len(separator)

        return -1

    def merge_chunks(self, chunks: List[DocumentChunk], max_length: int = 2000) -> List[DocumentChunk]:
        """合并小的文档块。"""
        if not chunks:
            return []

        merged_chunks = []
        current_chunk = DocumentChunk(
            content=chunks[0].content,
            metadata=chunks[0].metadata.copy(),
            doc_id=chunks[0].doc_id,
            chunk_index=chunks[0].chunk_index,
        )

        for next_chunk in chunks[1:]:
            combined_length = len(current_chunk.content) + len(next_chunk.content)

            if combined_length <= max_length and current_chunk.doc_id == next_chunk.doc_id:
                current_chunk.content += "\n" + next_chunk.content
                current_chunk.metadata["total_chunks"] = current_chunk.metadata.get("total_chunks", 1) + 1
            else:
                merged_chunks.append(current_chunk)
                current_chunk = DocumentChunk(
                    content=next_chunk.content,
                    metadata=next_chunk.metadata.copy(),
                    doc_id=next_chunk.doc_id,
                    chunk_index=next_chunk.chunk_index,
                )

        merged_chunks.append(current_chunk)

        log.info("DocumentProcessor merged | before={} after={}", len(chunks), len(merged_chunks))
        return merged_chunks

    def filter_chunks(self, chunks: List[DocumentChunk], min_length: int = 50) -> List[DocumentChunk]:
        """过滤太短的文档块。"""
        filtered = [chunk for chunk in chunks if len(chunk.content.strip()) >= min_length]
        log.info("DocumentProcessor filtered | before={} after={}", len(chunks), len(filtered))
        return filtered

    def add_chunk_metadata(self, chunks: List[DocumentChunk], metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """为文档块添加额外元数据。"""
        for chunk in chunks:
            chunk.metadata.update(metadata)
        return chunks


def load_text_file(file_path: str, encoding: str = "utf-8") -> Document:
    """加载文本文件为文档。"""
    with open(file_path, "r", encoding=encoding) as f:
        content = f.read()

    metadata = {
        "source": file_path,
        "type": "text_file",
        "loaded_at": datetime.now().isoformat(),
    }

    return Document(content=content, metadata=metadata, source=file_path)


def create_document(content: str, **metadata) -> Document:
    """创建文档的便捷函数。"""
    return Document(content=content, metadata=metadata)


__all__ = [
    "Document",
    "DocumentChunk",
    "DocumentProcessor",
    "load_text_file",
    "create_document",
]
