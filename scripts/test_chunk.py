"""测试 smart_chunk_markdown 是否会卡住"""
import re
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 模拟 OCR 输出：19000 chars 的中文文本
# 制造一些"恶意"格式：连续空白、没有空行的长段落
sample_paragraph = "大语言模型（LLM）是人工智能领域的核心技术。" * 50  # ~1500 chars
# 混入一些 OCR 常见的格式问题
evil_whitespace = " \t \t " * 100  # 大量空白但不是空行
ocr_text = ""
for i in range(15):
    ocr_text += f"# 第{i+1}章\n\n"
    ocr_text += sample_paragraph + "\n"
    ocr_text += evil_whitespace + "\n"  # 这里关键：\n + 空白 + \n
    ocr_text += sample_paragraph + "\n\n"

print(f"测试文本长度: {len(ocr_text)} chars")

# 测试1: re.split(r'\n\s*\n', text)
print("测试 re.split(r'\\n\\s*\\n', text)...")
t0 = time.time()
parts = re.split(r'\n\s*\n', ocr_text)
t1 = time.time()
print(f"  完成: {len(parts)} 部分, 耗时 {t1-t0:.3f}s")

# 测试2: 用 OCR 样本文件
sample_file = "/tmp/ocr_sample.md"
if os.path.exists(sample_file):
    with open(sample_file) as f:
        real_ocr = f.read()
    # 复制 10 倍模拟 10 页
    big_text = (real_ocr + "\n\n") * 10
    print(f"\n真实 OCR 文本 x10: {len(big_text)} chars")
    
    print("测试 re.split(r'\\n\\s*\\n', big_text)...")
    t0 = time.time()
    parts = re.split(r'\n\s*\n', big_text)
    t1 = time.time()
    print(f"  完成: {len(parts)} 部分, 耗时 {t1-t0:.3f}s")
    
    # 测试完整的 smart_chunk_markdown
    print("\n测试 smart_chunk_markdown...")
    from veragents.memory.rag.pipeline import smart_chunk_markdown, _preprocess_markdown_for_embedding
    t0 = time.time()
    chunks = smart_chunk_markdown(big_text)
    t1 = time.time()
    print(f"  完成: {len(chunks)} chunks, 耗时 {t1-t0:.3f}s")
    
    print("\n测试 _preprocess_markdown_for_embedding...")
    t0 = time.time()
    for c in chunks:
        _ = _preprocess_markdown_for_embedding(c["content"])
    t1 = time.time()
    print(f"  完成: 耗时 {t1-t0:.3f}s")

print("\n✅ 全部通过，没有卡住！")
