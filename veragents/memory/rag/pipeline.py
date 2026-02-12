"""RAG 管道完整实现。

五层七步架构：
- 用户层：RAGTool 统一接口
- 应用层：智能问答、搜索、管理
- 处理层：文档解析（MarkItDown）、Markdown 感知分块、向量化
- 存储层：Qdrant 向量数据库 + SQLite 文档存储
- 基础层：嵌入模型、LLM、数据库

处理流程：任意格式文档 → MarkItDown 转换 → Markdown 文本 → 智能分块 → 向量化 → 存储检索
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger as log

from veragents.memory.embedding import EmbeddingService, get_dimension, get_text_embedder
from veragents.memory.storage.document_store import SQLiteDocumentStore
from veragents.memory.storage.qdrant_store import QdrantStore

# ------------------------------------------------------------------ #
# MarkItDown 文档转换
# ------------------------------------------------------------------ #

_markitdown_instance = None


def _get_markitdown_instance():
    """延迟加载 MarkItDown 实例。"""
    global _markitdown_instance
    if _markitdown_instance is not None:
        return _markitdown_instance
    try:
        from markitdown import MarkItDown
        _markitdown_instance = MarkItDown()
        return _markitdown_instance
    except ImportError:
        log.warning("markitdown 未安装，使用回退文本读取器")
        return None


def _fallback_text_reader(path: str) -> str:
    """回退：纯文本读取。"""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        log.warning("回退读取失败: {} | error={}", path, e)
        return ""


def _enhanced_pdf_processing(path: str, batch_size: int = 10):
    """PDF 处理（流式生成器）：仅使用 MarkItDown[pdf]。"""
    _ = batch_size  # 兼容旧调用签名，PDF 现为一次性转换。

    md = _get_markitdown_instance()
    if md is None:
        log.error("[RAG] markitdown 未安装，无法处理 PDF: {}", path)
        yield ""
        return

    try:
        result = md.convert(path)
        text = getattr(result, "text_content", None)
        if isinstance(text, str) and text.strip():
            log.info("[RAG] PDF MarkItDown 转换成功 | path={} chars={}", path, len(text))
            yield text
            return
        log.warning("[RAG] PDF MarkItDown 输出为空: {}", path)
    except Exception as e:
        log.error("[RAG] PDF MarkItDown 转换失败: {} | error={}", path, e)

    yield ""


def convert_to_markdown(path: str) -> Iterator[str]:
    """通用文档读取器（流式）：将任意格式文档转换为 Markdown 文本Iterator。

    支持格式：
    - 文档：PDF、Word、Excel、PowerPoint
    - 图像：JPG、PNG、GIF（通过 OCR）
    - 音频：MP3、WAV、M4A（通过转录）
    - 文本：TXT、CSV、JSON、XML、HTML
    - 代码：Python、JavaScript、Java 等
    """
    if not os.path.exists(path):
        log.error("[RAG] 文件不存在: {}", path)
        yield ""
        return

    ext = (os.path.splitext(path)[1] or "").lower()

    # PDF 仅使用 MarkItDown[pdf] 处理
    if ext == ".pdf":
        yield from _enhanced_pdf_processing(path)
        return

    # 其他格式使用 MarkItDown 统一转换 (一次性 yield)
    md_instance = _get_markitdown_instance()
    if md_instance is None:
        yield _fallback_text_reader(path)
        return

    try:
        result = md_instance.convert(path)
        markdown_text = getattr(result, "text_content", None)
        if isinstance(markdown_text, str) and markdown_text.strip():
            log.info("[RAG] MarkItDown 转换成功: {} -> {} chars Markdown", path, len(markdown_text))
            yield markdown_text
            return
        yield ""
    except Exception as e:
        log.warning("[RAG] MarkItDown 转换失败 {}: {}", path, e)
        yield _fallback_text_reader(path)


# ------------------------------------------------------------------ #
# Token 估算（中英文混合）
# ------------------------------------------------------------------ #


def _is_cjk(ch: str) -> bool:
    """判断是否为 CJK 字符。"""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF      # CJK 统一汉字
        or 0x3400 <= code <= 0x4DBF    # CJK 扩展 A
        or 0x20000 <= code <= 0x2A6DF  # CJK 扩展 B
        or 0x2A700 <= code <= 0x2B73F  # CJK 扩展 C
        or 0x2B740 <= code <= 0x2B81F  # CJK 扩展 D
        or 0x2B820 <= code <= 0x2CEAF  # CJK 扩展 E
        or 0xF900 <= code <= 0xFAFF    # CJK 兼容汉字
    )


def _approx_token_len(text: str) -> int:
    """近似估计 Token 长度，支持中英文混合。"""
    cjk = sum(1 for ch in text if _is_cjk(ch))
    non_cjk_tokens = len([t for t in text.split() if t])
    return cjk + non_cjk_tokens


# ------------------------------------------------------------------ #
# Markdown 感知智能分块
# ------------------------------------------------------------------ #


_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")


def _clean_heading_title(title: str, max_chars: int = 80) -> str:
    """清洗 OCR 产出的标题文本，避免脏标题进入 heading_path。"""
    clean = (title or "").replace("\u200b", " ")
    clean = re.sub(r"^[#>\-\*\s]+", "", clean)
    clean = re.sub(r"[`*_~]", "", clean)
    clean = " ".join(clean.split()).strip(" .,:;|")
    if not clean:
        return ""
    if len(clean) > max_chars:
        return ""
    if _approx_token_len(clean) > 24:
        return ""
    return clean


def _normalize_heading_path(path: str, max_chars: int = 96) -> str:
    """压缩和规范 heading_path，便于展示和提示词使用。"""
    if not path:
        return ""
    parts = [_clean_heading_title(p, max_chars=max_chars) for p in path.split(" / ")]
    parts = [p for p in parts if p]
    if not parts:
        return ""
    merged = " / ".join(parts)
    if len(merged) > max_chars:
        return merged[: max_chars - 3].rstrip() + "..."
    return merged


def _split_paragraphs_with_headings(text: str) -> List[Dict]:
    """根据标题层次分割段落，保持语义完整性。"""
    lines = text.splitlines()
    heading_stack: List[str] = []
    paragraphs: List[Dict] = []
    buf: List[str] = []
    char_pos = 0

    def flush_buf(end_pos: int):
        if not buf:
            return
        content = "\n".join(buf).strip()
        if not content:
            return
        heading_path = _normalize_heading_path(" / ".join(heading_stack))
        paragraphs.append({
            "content": content,
            "heading_path": heading_path or None,
            "start": max(0, end_pos - len(content)),
            "end": end_pos,
        })

    for ln in lines:
        raw = ln
        heading_match = _HEADING_RE.match(raw)
        if heading_match:
            level = len(heading_match.group(1)) or 1
            title = _clean_heading_title(heading_match.group(2))
            if not title:
                buf.append(raw)
                char_pos += len(raw) + 1
                continue

            flush_buf(char_pos)
            buf = []

            if level <= len(heading_stack):
                heading_stack = heading_stack[:level - 1]
            heading_stack.append(title)

            char_pos += len(raw) + 1
            continue

        # 段落内容累积
        if raw.strip() == "":
            flush_buf(char_pos)
            buf = []
        else:
            buf.append(raw)
        char_pos += len(raw) + 1

    flush_buf(char_pos)

    if not paragraphs:
        paragraphs = [{"content": text, "heading_path": None, "start": 0, "end": len(text)}]

    return paragraphs


def _chunk_paragraphs(
    paragraphs: List[Dict],
    chunk_tokens: int = 512,
    overlap_tokens: int = 64,
) -> List[Dict]:
    """基于 Token 数量的智能分块，支持重叠策略并保证循环可前进。"""
    if chunk_tokens <= 0:
        log.warning("[RAG] chunk_tokens={} 非法，自动回退为 1", chunk_tokens)
        chunk_tokens = 1

    overlap_tokens = max(0, overlap_tokens)
    if overlap_tokens >= chunk_tokens:
        safe_overlap = max(chunk_tokens // 4, 0)
        log.warning(
            "[RAG] overlap_tokens={} >= chunk_tokens={}，自动调整为 {}",
            overlap_tokens,
            chunk_tokens,
            safe_overlap,
        )
        overlap_tokens = safe_overlap

    normalized: List[Dict] = []
    for p in paragraphs:
        content = (p.get("content") or "").strip()
        if not content:
            continue
        item = dict(p)
        item["content"] = content
        item["_tokens"] = _approx_token_len(content) or 1
        normalized.append(item)

    chunks: List[Dict] = []
    cur: List[Dict] = []
    cur_tokens = 0
    i = 0

    def _append_chunk(items: List[Dict]) -> None:
        if not items:
            return
        content = "\n\n".join(x["content"] for x in items)
        start = items[0].get("start", 0)
        end = items[-1].get("end", start + len(content))
        heading_path = next(
            (x.get("heading_path") for x in reversed(items) if x.get("heading_path")),
            None,
        )
        chunks.append({
            "content": content,
            "start": start,
            "end": end,
            "heading_path": heading_path,
        })

    while i < len(normalized):
        p = normalized[i]
        p_tokens = p["_tokens"]

        # 超长段落：先落盘当前块，再对超长段落做硬切分
        if p_tokens > chunk_tokens:
            if cur:
                _append_chunk(cur)
                cur = []
                cur_tokens = 0
                continue

            log.warning("[RAG] 遇到超长段落 ({} tokens)，强行切分", p_tokens)
            sub_chunks = _split_long_text(p["content"], chunk_tokens, overlap_tokens)
            for sub in sub_chunks:
                chunks.append({
                    "content": sub,
                    "start": p.get("start", 0),
                    "end": p.get("end", len(p["content"])),
                    "heading_path": p.get("heading_path"),
                })
            i += 1
            continue

        if cur_tokens + p_tokens <= chunk_tokens:
            cur.append(p)
            cur_tokens += p_tokens
            i += 1
            continue

        _append_chunk(cur)

        if overlap_tokens > 0 and cur:
            kept: List[Dict] = []
            kept_tokens = 0
            for x in reversed(cur):
                t = x["_tokens"]
                if kept_tokens + t > overlap_tokens:
                    break
                kept.append(x)
                kept_tokens += t
            cur = list(reversed(kept))
            cur_tokens = kept_tokens
        else:
            cur = []
            cur_tokens = 0

        # 防止 overlap 过大导致同一段落反复无法插入。
        if cur and cur_tokens + p_tokens > chunk_tokens:
            cur = []
            cur_tokens = 0

    _append_chunk(cur)
    return chunks


def _split_long_text(text: str, chunk_tokens: int, overlap_tokens: int) -> List[str]:
    """将超长文本切分为多个分块。"""
    chunks = []
    start = 0
    text_len = len(text)
    token_len = _approx_token_len(text)
    
    if token_len <= chunk_tokens:
        return [text]
    
    if token_len == 0:
        return [text]
        
    # 简单按字符比例估算切分
    chunk_chars = max(int(chunk_tokens * (text_len / token_len)), 100)  # 至少 100 字符
    overlap_chars = int(overlap_tokens * (text_len / token_len))
    
    # 安全保障：overlap 不能超过 chunk 的一半，否则 start 无法前进
    overlap_chars = min(overlap_chars, chunk_chars // 2)
    
    max_chunks = (text_len // max(chunk_chars - overlap_chars, 1)) + 2  # 安全上限
    
    while start < text_len and len(chunks) < max_chunks:
        end = min(text_len, start + chunk_chars)
        # 尽量在换行符切分
        if end < text_len:
            last_nl = text.rfind('\n', start, end)
            if last_nl > start + chunk_chars // 2:
                end = last_nl + 1
            
        chunks.append(text[start:end])
        
        # 确保 start 始终前进（至少前进 1 个字符）
        next_start = end - overlap_chars
        if next_start <= start:
            next_start = start + max(chunk_chars // 2, 1)
        start = next_start
        
    return chunks


def smart_chunk_markdown(
    text: str,
    chunk_tokens: int = 512,
    overlap_tokens: int = 64,
    min_chunk_chars: int = 50,
) -> List[Dict]:
    """Markdown 感知的智能分块入口。

    流程：标准 Markdown → 标题层次解析 → 段落语义分割 → Token 计算分块 → 重叠策略优化
    """
    if not text or not text.strip():
        return []

    # 1. 线性段落解析，避免 re.split(r"\n\s*\n", ...) 在噪声 OCR 文本中的退化。
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = _split_paragraphs_with_headings(normalized_text)

    # 2. 基于 Token 数量分块 + 重叠
    chunks = _chunk_paragraphs(paragraphs, chunk_tokens, overlap_tokens)

    # 3. 过滤太短的块
    chunks = [c for c in chunks if len(c["content"].strip()) >= min_chunk_chars]

    # 4. 为每个块添加索引 ID
    for idx, chunk in enumerate(chunks):
        chunk["chunk_index"] = idx
        chunk["token_estimate"] = _approx_token_len(chunk["content"])

    log.info(
        "[RAG] 智能分块完成 | paragraphs={} chunks={} total_tokens={}",
        len(paragraphs),
        len(chunks),
        sum(c["token_estimate"] for c in chunks),
    )
    return chunks


# ------------------------------------------------------------------ #
# Markdown 预处理（嵌入前优化）
# ------------------------------------------------------------------ #


def _preprocess_markdown_for_embedding(text: str) -> str:
    """预处理 Markdown 文本以获得更好的嵌入质量。
    注意：避免使用可能产生灾难性回溯的正则表达式。
    """
    import re
    # 只做安全的、不会回溯的简单替换
    text = re.sub(r"#{1,6}\s*", "", text)       # 去掉标题标记
    text = re.sub(r"\n{3,}", "\n\n", text)       # 压缩连续空行
    text = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", text)  # 图片（用[^]避免回溯）
    text = re.sub(r"\[[^\]]*\]\([^\)]*\)", "", text)   # 链接
    # 不再尝试匹配 *bold*/`code` 等，这些正则在 OCR 输出上容易 ReDoS
    return text.strip()


# ------------------------------------------------------------------ #
# 嵌入辅助
# ------------------------------------------------------------------ #


def embed_query(query: str) -> List[float]:
    """嵌入单个查询文本。"""
    embedder = get_text_embedder()
    vec = embedder.encode(query)
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    return list(vec)


# ------------------------------------------------------------------ #
# 向量索引
# ------------------------------------------------------------------ #


def index_chunks(
    store: Optional[QdrantStore] = None,
    chunks: Optional[List[Dict]] = None,
    doc_store: Optional[SQLiteDocumentStore] = None,
    batch_size: int = 16,
    rag_namespace: str = "default",
    source: str = "",
) -> int:
    """将 Markdown 分块向量化并索引到 Qdrant + SQLite。

    Returns:
        成功索引的分块数量。
    """
    if not chunks:
        log.info("[RAG] 没有需要索引的分块")
        return 0

    embedder = get_text_embedder()
    dimension = get_dimension(384)

    # 创建默认 Qdrant 存储
    if store is None:
        store = _create_default_vector_store(dimension)

    # 预处理 Markdown 文本用于嵌入
    processed_texts = [_preprocess_markdown_for_embedding(c["content"]) for c in chunks]

    total_batches = (len(processed_texts) + batch_size - 1) // batch_size
    log.info("[RAG] 开始嵌入 | total={} batch_size={} batches={}", len(processed_texts), batch_size, total_batches)

    total_indexed = 0

    for i in range(0, len(processed_texts), batch_size):
        batch_num = i // batch_size + 1
        batch_texts = processed_texts[i:i + batch_size]
        batch_chunks = chunks[i:i + batch_size]
        log.info("[RAG] 嵌入 batch {}/{} ({}条)...", batch_num, total_batches, len(batch_texts))

        try:
            # 批量编码
            raw_vecs = embedder.encode(batch_texts)

            # 标准化为 List[List[float]]
            vecs: List[List[float]] = []
            if not isinstance(raw_vecs, list):
                if hasattr(raw_vecs, "tolist"):
                    vecs = [raw_vecs.tolist()]
                else:
                    vecs = [list(raw_vecs)]
            else:
                for v in raw_vecs:
                    if hasattr(v, "tolist"):
                        v = v.tolist()
                    v_norm = [float(x) for x in v]
                    # 维度检查和调整
                    if len(v_norm) != dimension:
                        log.warning("[RAG] 向量维度异常: 期望 {} 实际 {}", dimension, len(v_norm))
                        if len(v_norm) < dimension:
                            v_norm.extend([0.0] * (dimension - len(v_norm)))
                        else:
                            v_norm = v_norm[:dimension]
                    vecs.append(v_norm)

            # 构建元数据并写入 Qdrant
            metadata_list: List[Dict[str, Any]] = []
            ids: List[str] = []
            for idx, chunk in enumerate(batch_chunks):
                chunk_id = str(uuid.uuid4())
                ids.append(chunk_id)
                metadata_list.append({
                    "memory_id": chunk_id,
                    "memory_type": "rag_chunk",
                    "content": chunk["content"][:500],
                    "heading_path": chunk.get("heading_path", ""),
                    "chunk_index": chunk.get("chunk_index", i + idx),
                    "source": source,
                    "rag_namespace": rag_namespace,
                    "is_rag_data": True,
                    "data_source": "rag_pipeline",
                    "timestamp": int(datetime.now().timestamp()),
                })

            store.add_vectors(vectors=vecs, metadata=metadata_list, ids=ids)

            # 同步写入 SQLite 文档存储
            if doc_store is not None:
                for chunk_id, chunk_meta, bc in zip(ids, metadata_list, batch_chunks):
                    doc_store.add_memory(
                        memory_id=chunk_id,
                        user_id="rag_system",
                        content=bc["content"],
                        memory_type="rag_chunk",
                        timestamp=chunk_meta["timestamp"],
                        importance=0.5,
                        properties={
                            "heading_path": bc.get("heading_path", ""),
                            "chunk_index": bc.get("chunk_index", 0),
                            "source": source,
                            "rag_namespace": rag_namespace,
                        },
                    )

            total_indexed += len(vecs)
            log.info("[RAG] 索引进度: {}/{}", min(i + batch_size, len(chunks)), len(chunks))

        except Exception as e:
            log.error("[RAG] 批次 {} 索引失败: {}", i, e)

    log.info("[RAG] 索引完成 | total_indexed={}", total_indexed)
    return total_indexed


def _create_default_vector_store(dimension: Optional[int] = None) -> QdrantStore:
    """创建默认的 Qdrant 向量存储。"""
    if dimension is None:
        dimension = get_dimension(384)
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection = os.getenv("QDRANT_COLLECTION", "veragents_vectors")
    return QdrantStore(
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection=f"{collection}_rag",
        vector_size=dimension,
    )


# ------------------------------------------------------------------ #
# 高级检索策略
# ------------------------------------------------------------------ #

_ZH_QUERY_STOPWORDS = {
    "什么",
    "什么是",
    "请问",
    "如何",
    "怎么",
    "为什么",
    "定义",
    "介绍",
    "解释",
    "一下",
    "一下子",
    "原理",
    "概念",
}


def _heuristic_query_expansions(query: str, limit: int = 3) -> List[str]:
    """规则化查询扩展，优先改善“什么是 X”这类定义型问题。"""
    q = (query or "").strip()
    if not q:
        return []

    candidates: List[str] = []
    m = re.match(r"^(?:什么是|啥是|什么叫|请解释(?:一下)?|what is|what's|define)\s*[:：]?\s*(.+)$", q, flags=re.IGNORECASE)
    if m:
        topic = m.group(1).strip().strip("。！？?!,.")
        topic = re.sub(r"\s+", " ", topic)
        if topic:
            candidates.extend([
                topic,
                f"{topic} 定义",
                f"{topic} 原理",
                f"{topic} 作用",
            ])
    elif len(q) > 24:
        # 普通长查询不做规则扩展，避免增加无效检索开销。
        return []

    # 对短查询追加中英词面，增强对术语命中。
    ascii_terms = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", q)]
    zh_terms = [t for t in re.findall(r"[\u4e00-\u9fff]{2,}", q) if t not in _ZH_QUERY_STOPWORDS]
    for term in ascii_terms + zh_terms:
        candidates.append(term)

    uniq: List[str] = []
    for c in candidates:
        c = c.strip()
        if c and c not in uniq:
            uniq.append(c)
    return uniq[:limit]


def _extract_query_terms(query: str) -> List[str]:
    """提取用于轻量重排的关键词。"""
    q = (query or "").strip()
    if not q:
        return []

    terms: List[str] = []
    terms.extend([t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", q)])
    terms.extend([t for t in re.findall(r"[\u4e00-\u9fff]{2,}", q) if t not in _ZH_QUERY_STOPWORDS])
    m = re.match(r"^(?:什么是|啥是|什么叫|请解释(?:一下)?|what is|what's|define)\s*[:：]?\s*(.+)$", q, flags=re.IGNORECASE)
    if m:
        topic = m.group(1).strip().strip("。！？?!,.").lower()
        topic = re.sub(r"\s+", " ", topic)
        if topic:
            terms.append(topic)

    uniq: List[str] = []
    for t in terms:
        t = t.strip()
        if t and t not in uniq:
            uniq.append(t)
    return uniq[:10]


def _lexical_hit_score(text: str, terms: List[str]) -> float:
    """关键词命中得分 [0, 1]。"""
    if not text or not terms:
        return 0.0

    content = text.lower()
    matched = 0
    for term in terms:
        needle = term.lower()
        if needle and needle in content:
            matched += 1
    return matched / max(len(terms), 1)


def _prompt_mqe(query: str, n: int) -> List[str]:
    """使用 LLM 生成多样化的查询扩展（Multi-Query Expansion）。"""
    try:
        from veragents.core.llm import LLMClient
        from veragents.core.messages import Message
        llm = LLMClient()
        messages = [
            Message(content="你是检索查询扩展助手。生成语义等价或互补的多样化查询。使用中文，简短，避免标点。", role="system"),
            Message(content=f"原始查询：{query}\n请给出{n}个不同表述的查询，每行一个。", role="user"),
        ]
        text = llm.chat(messages)
        lines = [ln.strip("- \t0123456789.、") for ln in (text or "").splitlines()]
        outs = [ln for ln in lines if ln and len(ln) > 1]
        return outs[:n] or [query]
    except Exception as e:
        log.warning("[RAG] MQE 扩展失败: {}", e)
        return [query]


def _prompt_hyde(query: str) -> Optional[str]:
    """生成假设性文档用于改善检索（HyDE）。"""
    try:
        from veragents.core.llm import LLMClient
        from veragents.core.messages import Message
        llm = LLMClient()
        messages = [
            Message(
                content="根据用户问题，先写一段可能的答案性段落，用于向量检索的查询文档（不要分析过程）。",
                role="system",
            ),
            Message(
                content=f"问题：{query}\n请直接写一段中等长度、客观、包含关键术语的段落。",
                role="user",
            ),
        ]
        return llm.chat(messages)
    except Exception as e:
        log.warning("[RAG] HyDE 生成失败: {}", e)
        return None


def search_vectors(
    store: Optional[QdrantStore] = None,
    query: str = "",
    top_k: int = 8,
    rag_namespace: Optional[str] = None,
    score_threshold: Optional[float] = None,
    enable_mqe: bool = False,
    mqe_expansions: int = 2,
    enable_hyde: bool = False,
    candidate_pool_multiplier: int = 4,
) -> List[Dict]:
    """带扩展策略的向量检索。

    Args:
        store: Qdrant 存储实例
        query: 查询文本
        top_k: 返回结果数量
        rag_namespace: RAG 命名空间过滤
        score_threshold: 分数阈值过滤
        enable_mqe: 启用多查询扩展
        mqe_expansions: MQE 扩展数量
        enable_hyde: 启用假设文档嵌入
        candidate_pool_multiplier: 候选池倍数

    Returns:
        List[Dict]: 排序后的检索结果
    """
    if not query:
        return []

    if store is None:
        store = _create_default_vector_store()

    # 查询扩展
    expansions: List[str] = [query]
    heuristic_expansions = _heuristic_query_expansions(query)
    if heuristic_expansions:
        expansions.extend(heuristic_expansions)

    if enable_mqe and mqe_expansions > 0:
        mqe_results = _prompt_mqe(query, mqe_expansions)
        expansions.extend(mqe_results)
        log.info("[RAG] MQE 扩展: {} -> {}", query[:30], mqe_results)

    if enable_hyde:
        hyde_text = _prompt_hyde(query)
        if hyde_text:
            expansions.append(hyde_text)
            log.info("[RAG] HyDE 生成假设文档: {} chars", len(hyde_text))

    # 去重
    uniq: List[str] = []
    for e in expansions:
        if e and e not in uniq:
            uniq.append(e)
    expansions = uniq

    # 分配候选池
    pool = max(top_k * candidate_pool_multiplier, 20)
    per = max(1, pool // max(1, len(expansions)))

    # 构建过滤器
    where: Dict[str, Any] = {"memory_type": "rag_chunk"}
    if rag_namespace:
        where["rag_namespace"] = rag_namespace

    # 收集所有扩展查询的结果
    agg: Dict[str, Dict] = {}
    for q in expansions:
        qv = embed_query(q)
        hits = store.search_similar(
            query_vector=qv,
            limit=per,
            score_threshold=score_threshold,
            where=where,
        )
        for h in hits:
            mid = h.get("metadata", {}).get("memory_id", h.get("id"))
            s = float(h.get("score", 0.0))
            if mid not in agg or s > float(agg[mid].get("score", 0.0)):
                agg[mid] = h

    # 按分数排序返回
    merged = list(agg.values())
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

    log.info("[RAG] 检索完成 | expansions={} candidates={} returned={}", len(expansions), len(agg), min(len(merged), top_k))
    return merged[:top_k]


# ------------------------------------------------------------------ #
# RAG Pipeline 主类
# ------------------------------------------------------------------ #


class RAGPipeline:
    """端到端的检索增强生成管道。

    处理链路：文档 → MarkItDown 转换 → Markdown 分块 → 向量化 → Qdrant 存储 → 检索 → LLM 增强

    Args:
        knowledge_base_path: 知识库根目录
        qdrant_url: Qdrant 服务地址
        qdrant_api_key: Qdrant API Key
        collection_name: 向量集合名称
        rag_namespace: 命名空间（多租户隔离）
        chunk_tokens: 分块 Token 数
        overlap_tokens: 重叠 Token 数
    """

    def __init__(
        self,
        knowledge_base_path: str = "./knowledge_base",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "rag_knowledge_base",
        rag_namespace: str = "default",
        chunk_tokens: int = 512,
        overlap_tokens: int = 64,
    ):
        self.knowledge_base_path = knowledge_base_path
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name
        self.rag_namespace = rag_namespace
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens

        os.makedirs(knowledge_base_path, exist_ok=True)

        # 嵌入模型
        self.embedder = get_text_embedder()
        self.dimension = get_dimension(384)

        # 向量存储
        self.vector_store = QdrantStore(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            collection=self.collection_name,
            vector_size=self.dimension,
        )

        # 文档存储
        db_path = os.path.join(knowledge_base_path, "rag_docs.db")
        self.doc_store = SQLiteDocumentStore(db_path=db_path)

        # LLM（延迟初始化）
        self._llm = None

        # 统计信息
        self._indexed_files: List[str] = []
        self._total_chunks: int = 0

        log.info(
            "[RAG] Pipeline initialized | kb_path={} collection={} namespace={} dim={}",
            knowledge_base_path,
            self.collection_name,
            self.rag_namespace,
            self.dimension,
        )

    @property
    def llm(self):
        if self._llm is None:
            from veragents.core.llm import LLMClient
            self._llm = LLMClient()
        return self._llm

    # ------------------------------------------------------------------ #
    # 文档载入
    # ------------------------------------------------------------------ #

    def ingest_file(self, file_path: str) -> int:
        """载入单个文件到知识库（流式）。
        
        流程：文件 → MarkItDown 转换(Stream) → 智能分块 → 向量化 → 存储
        """
        log.info("[RAG] 载入文件: {}", file_path)
        
        total_count = 0
        has_content = False
        batch_num = 0

        # 流式消费，每次拿到一部分文档内容（例如 PDF 的 10 页）
        for markdown_text in convert_to_markdown(file_path):
            if not markdown_text or not markdown_text.strip():
                continue
            
            batch_num += 1
            has_content = True
            log.info("[RAG] ingest_file 收到批次 {} | {} chars", batch_num, len(markdown_text))
            
            # 2. 智能分块
            log.info("[RAG] 开始 smart_chunk_markdown...")
            chunks = smart_chunk_markdown(
                markdown_text,
                chunk_tokens=self.chunk_tokens,
                overlap_tokens=self.overlap_tokens,
            )
            log.info("[RAG] smart_chunk_markdown 完成 | {} chunks", len(chunks) if chunks else 0)

            if not chunks:
                continue

            # 3. 向量化 + 存储 (当前批次)
            log.info("[RAG] 开始 index_chunks...")
            count = index_chunks(
                store=self.vector_store,
                chunks=chunks,
                doc_store=self.doc_store,
                rag_namespace=self.rag_namespace,
                source=file_path,
            )
            total_count += count
            log.info("[RAG] index_chunks 完成 | count={}", count)
            
            # 显式清理内存
            del chunks
            import gc
            gc.collect()

        if not has_content:
            log.warning("[RAG] 文件处理为空: {}", file_path)
            return 0

        self._indexed_files.append(file_path)
        self._total_chunks += total_count

        log.info("[RAG] 文件载入完成: {} | total_chunks={}", file_path, total_count)
        return total_count

    def ingest_text(
        self,
        text: str,
        source: str = "direct_input",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """直接载入文本到知识库。"""
        log.info("[RAG] 载入文本 | source={} len={}", source, len(text))

        chunks = smart_chunk_markdown(
            text,
            chunk_tokens=self.chunk_tokens,
            overlap_tokens=self.overlap_tokens,
        )

        if not chunks:
            return 0

        count = index_chunks(
            store=self.vector_store,
            chunks=chunks,
            doc_store=self.doc_store,
            rag_namespace=self.rag_namespace,
            source=source,
        )

        self._total_chunks += count
        return count

    def ingest_directory(self, dir_path: str, extensions: Optional[List[str]] = None) -> int:
        """递归载入目录下的所有文件。"""
        if extensions is None:
            extensions = [
                ".txt", ".md", ".pdf", ".docx", ".doc",
                ".xlsx", ".xls", ".pptx", ".ppt",
                ".csv", ".json", ".xml", ".html", ".htm",
                ".py", ".js", ".java", ".go", ".rs", ".c", ".cpp",
            ]

        total = 0
        for root, _dirs, files in os.walk(dir_path):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in extensions:
                    fpath = os.path.join(root, fname)
                    try:
                        count = self.ingest_file(fpath)
                        total += count
                    except Exception as e:
                        log.error("[RAG] 载入失败: {} | error={}", fpath, e)

        log.info("[RAG] 目录载入完成 | dir={} total_chunks={}", dir_path, total)
        return total

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        enable_mqe: bool = False,
        mqe_expansions: int = 2,
        enable_hyde: bool = False,
    ) -> List[Dict[str, Any]]:
        """检索知识库。

        Args:
            query: 查询文本
            top_k: 返回数量
            score_threshold: 分数阈值
            enable_mqe: 启用多查询扩展
            mqe_expansions: MQE 扩展数量
            enable_hyde: 启用假设文档嵌入

        Returns:
            List[Dict]: 检索结果列表，包含 content、score、metadata
        """
        candidate_k = max(top_k * 4, top_k)
        raw_results = search_vectors(
            store=self.vector_store,
            query=query,
            top_k=candidate_k,
            rag_namespace=self.rag_namespace,
            score_threshold=score_threshold,
            enable_mqe=enable_mqe,
            mqe_expansions=mqe_expansions,
            enable_hyde=enable_hyde,
        )

        # 从 doc_store 获取完整内容
        results: List[Dict[str, Any]] = []
        for hit in raw_results:
            meta = hit.get("metadata", {})
            memory_id = meta.get("memory_id", hit.get("id"))
            score = float(hit.get("score", 0.0))

            # 尝试从 SQLite 获取完整内容
            doc = self.doc_store.get_memory(memory_id) if memory_id else None
            if doc:
                content = doc["content"]
                properties = doc.get("properties", {}) or {}
            else:
                content = meta.get("content", "")
                properties = {}

            results.append({
                "id": memory_id,
                "content": content,
                "score": score,
                "heading_path": _normalize_heading_path(
                    (meta.get("heading_path") or properties.get("heading_path", "") or "")
                ),
                "source": meta.get("source") or properties.get("source", ""),
                "chunk_index": meta.get("chunk_index", properties.get("chunk_index", 0)),
                "namespace": meta.get("rag_namespace", self.rag_namespace),
            })

        # 轻量关键词重排：保留向量分，同时提升术语精确匹配片段。
        terms = _extract_query_terms(query)
        lexical_weight = 0.18 if len(query.strip()) <= 24 else 0.12
        for item in results:
            heading = item.get("heading_path", "")
            content = item.get("content", "")
            lexical_text = f"{heading}\n{content[:2000]}"
            lexical_score = _lexical_hit_score(lexical_text, terms)
            base = float(item.get("score", 0.0))
            item["lexical_score"] = lexical_score
            item["rank_score"] = base + lexical_weight * lexical_score

        results.sort(key=lambda x: float(x.get("rank_score", x.get("score", 0.0))), reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------ #
    # LLM 增强问答
    # ------------------------------------------------------------------ #

    def query(
        self,
        question: str,
        top_k: int = 5,
        enable_mqe: bool = False,
        enable_hyde: bool = False,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """执行 RAG 问答：检索 → 拼接上下文 → LLM 生成。

        Args:
            question: 用户问题
            top_k: 检索数量
            enable_mqe: 启用多查询扩展
            enable_hyde: 启用假设文档嵌入
            system_prompt: 自定义系统提示词

        Returns:
            Dict: 包含 answer、sources、context
        """
        # 1. 检索
        search_results = self.search(
            query=question,
            top_k=top_k,
            enable_mqe=enable_mqe,
            enable_hyde=enable_hyde,
        )

        if not search_results:
            return {
                "answer": "未找到相关知识。请检查知识库是否已载入相关文档。",
                "sources": [],
                "context": "",
            }

        # 2. 构建上下文
        context_parts = []
        for i, result in enumerate(search_results, 1):
            heading = _normalize_heading_path(result.get("heading_path", ""))
            heading_str = f"（{heading}）" if heading else ""
            source = result.get("source", "")
            source_str = f" [来源: {os.path.basename(source)}]" if source else ""
            context_parts.append(f"【参考 {i}】{heading_str}{source_str}\n{result['content']}")

        context = "\n\n---\n\n".join(context_parts)

        # 3. 构建 Prompt
        if system_prompt is None:
            system_prompt = (
                "你是一个基于知识库的智能问答助手。请根据提供的参考内容回答用户问题。\n"
                "要求：\n"
                "1. 仅基于参考内容回答，不要编造信息\n"
                "2. 如果参考内容不足以回答问题，请如实说明\n"
                "3. 回答要准确、简洁、有条理\n"
                "4. 适当引用参考来源"
            )

        from veragents.core.messages import Message
        messages = [
            Message(content=system_prompt, role="system"),
            Message(
                content=f"参考内容：\n{context}\n\n用户问题：{question}",
                role="user",
            ),
        ]

        # 4. LLM 生成
        try:
            answer = self.llm.chat(messages)
        except Exception as e:
            log.error("[RAG] LLM 生成失败: {}", e)
            answer = f"LLM 调用失败: {e}\n\n参考内容已检索到 {len(search_results)} 条结果。"

        return {
            "answer": answer,
            "sources": search_results,
            "context": context,
        }

    # ------------------------------------------------------------------ #
    # 管理
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict[str, Any]:
        """获取 RAG 管道统计信息。"""
        try:
            vs_stats = self.vector_store.get_collection_stats()
        except Exception:
            vs_stats = {}
        db_stats = self.doc_store.get_database_stats()

        return {
            "namespace": self.rag_namespace,
            "collection": self.collection_name,
            "indexed_files": len(self._indexed_files),
            "total_chunks": self._total_chunks,
            "vector_store": vs_stats,
            "document_store": {
                k: v for k, v in db_stats.items()
                if k.endswith("_count") or k in ["store_type", "db_path"]
            },
            "dimension": self.dimension,
            "chunk_config": {
                "chunk_tokens": self.chunk_tokens,
                "overlap_tokens": self.overlap_tokens,
            },
        }

    def clear(self) -> None:
        """清空当前命名空间的所有 RAG 数据。"""
        # 清空 SQLite 中的 rag_chunk 记录
        docs = self.doc_store.search_memories(memory_type="rag_chunk", limit=100000)
        ids = [d["memory_id"] for d in docs]
        for mid in ids:
            self.doc_store.delete_memory(mid)

        # 清空 Qdrant 向量
        try:
            if ids:
                self.vector_store.delete_memories(ids)
        except Exception as e:
            log.warning("[RAG] Qdrant 清空失败: {}", e)

        self._indexed_files.clear()
        self._total_chunks = 0
        log.info("[RAG] 管道已清空 | deleted={}", len(ids))


# ------------------------------------------------------------------ #
# 便捷工厂函数
# ------------------------------------------------------------------ #


def create_rag_pipeline(
    knowledge_base_path: str = "./knowledge_base",
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    collection_name: str = "rag_knowledge_base",
    rag_namespace: str = "default",
    chunk_tokens: int = 512,
    overlap_tokens: int = 64,
) -> RAGPipeline:
    """创建 RAG 管道的工厂函数。"""
    return RAGPipeline(
        knowledge_base_path=knowledge_base_path,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection_name,
        rag_namespace=rag_namespace,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
    )


__all__ = [
    "RAGPipeline",
    "create_rag_pipeline",
    "convert_to_markdown",
    "smart_chunk_markdown",
    "index_chunks",
    "search_vectors",
    "embed_query",
]
