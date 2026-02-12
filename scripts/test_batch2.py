"""OCR 第 11-20 页并保存，然后测试分块"""
import os, sys, fitz, base64, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI

client = OpenAI(api_key=os.getenv("AIPING_API_KEY"), base_url=os.getenv("AIPING_BASE_URL","https://aiping.cn/api/v1"))

pdf_path = "/Users/nev8r/Downloads/Happy-LLM-0727.pdf"
doc = fitz.open(pdf_path)

results = []
for page_idx in range(10, 20):  # 第 11-20 页
    page = doc.load_page(page_idx)
    pix = page.get_pixmap(dpi=72)
    img = pix.tobytes("jpg")
    b64 = base64.b64encode(img).decode()
    del pix, page
    
    print(f"OCR 第 {page_idx+1} 页...", end=" ", flush=True)
    resp = client.chat.completions.create(
        model="DeepSeek-OCR",
        messages=[{"role":"user","content":[
            {"type":"text","text":"OCR this image to Markdown."},
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
        ]}], max_tokens=4096)
    text = resp.choices[0].message.content
    results.append(text)
    print(f"{len(text)} chars")
    del b64, img, resp

doc.close()
gc.collect()

# 合并并保存
full_text = "\n\n".join(results)
with open("/tmp/ocr_batch2.txt", "w") as f:
    f.write(full_text)
print(f"\n合并后: {len(full_text)} chars, 保存到 /tmp/ocr_batch2.txt")

# 现在测试分块
print("\n===== 测试分块 =====")
from veragents.memory.rag.pipeline import smart_chunk_markdown, _preprocess_markdown_for_embedding

print("smart_chunk_markdown...", flush=True)
t0 = time.time()
chunks = smart_chunk_markdown(full_text)
t1 = time.time()
print(f"  完成: {len(chunks)} chunks, {t1-t0:.3f}s")

print("_preprocess_markdown_for_embedding...", flush=True)
t0 = time.time()
for c in chunks:
    _ = _preprocess_markdown_for_embedding(c["content"])
t1 = time.time()
print(f"  完成: {t1-t0:.3f}s")

print("\n✅ 全部通过！")
