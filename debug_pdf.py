from dotenv import load_dotenv
load_dotenv()

import sys
import os
from loguru import logger as log

# 调整日志输出
log.remove()
log.add(sys.stderr, level="INFO")

from veragents.memory.rag.pipeline import convert_to_markdown, smart_chunk_markdown

# 查找 gradio 缓存的 PDF（如果你刚才上传了，应该在这里）
import glob
pdfs = glob.glob('/private/var/folders/**/gradio/**/*.pdf', recursive=True)

if not pdfs:
    print("❌ 没有找到缓存的 PDF，请手动指定路径")
    sys.exit(1)

# 取最新的一个
target_pdf = max(pdfs, key=os.path.getmtime)
print(f"📄 诊断 PDF: {target_pdf}")

# 1. 转换测试
print("\n--- [Step 1: MarkItDown 转换] ---")
md_text = convert_to_markdown(target_pdf)
print(f"✅ 转换结果长度: {len(md_text)}")
print(f"👀 前 500 字符预览:\n{'-'*40}\n{md_text[:500]}\n{'-'*40}")

# 检查是否包含 Markdown 标题
if "# " in md_text:
    print("✅ 检测到 Markdown 标题 (# )")
else:
    print("⚠️ 警告: 未检测到 Markdown 标题，可能是纯文本提取（PyPDF2回退）")

# 2. 分块测试
print("\n--- [Step 2: 智能分块] ---")
chunks = smart_chunk_markdown(md_text, chunk_tokens=512, overlap_tokens=64)
print(f"✅ 分块数量: {len(chunks)}")

if chunks:
    print("\n--- 🔎 抽查前 5 个分块 ---")
    for i, c in enumerate(chunks[:5]):
        content = c['content'].replace('\n', ' ')[:100]
        heading = c.get('heading_path', 'None')
        print(f"[{i}] 标题路径: {heading}")
        print(f"    内容: {content}...")
        
    print("\n--- 🔎 抽查包含 'Transformer' 的分块 ---")
    found = False
    for c in chunks:
        if "Transformer" in c['content']:
            print(f"🎯 找到相关分块 | 标题: {c.get('heading_path')}")
            print(f"   内容片段: {c['content'][:150].replace(chr(10), ' ')}...")
            found = True
            break
    if not found:
        print("❌ 未找到包含 'Transformer' 的分块")
else:
    print("❌ 分块结果为空！")
