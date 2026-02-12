"""极简 OCR 测试脚本：只看 1 页结果"""
import os, sys, fitz, base64
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI

client = OpenAI(api_key=os.getenv("AIPING_API_KEY"), base_url=os.getenv("AIPING_BASE_URL","https://aiping.cn/api/v1"))
doc = fitz.open("/Users/nev8r/Downloads/Happy-LLM-0727.pdf")

# 只看第5页
page = doc.load_page(4)
pix = page.get_pixmap(dpi=72)
img = pix.tobytes("jpg")
b64 = base64.b64encode(img).decode()
print(f"base64 size: {len(b64)} bytes")
del pix, page
doc.close()

resp = client.chat.completions.create(
    model="DeepSeek-OCR",
    messages=[{"role":"user","content":[
        {"type":"text","text":"OCR this image to Markdown."},
        {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
    ]}],
    max_tokens=4096
)
text = resp.choices[0].message.content
with open("/tmp/ocr_sample.md","w") as f:
    f.write(text)
print(f"OCR output: {len(text)} chars")
print("--- FULL OUTPUT ---")
print(text)
