#!/usr/bin/env python3
"""轻量级 PDF 入库脚本（不加载 Gradio，极省内存）。

用法：
    python scripts/ingest_pdf.py <pdf_path> [user_id]

示例：
    python scripts/ingest_pdf.py ~/Documents/Happy-LLM-0727.pdf user_test
"""
import sys
import os
import gc
import time

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/ingest_pdf.py <pdf_path> [user_id]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    user_id = sys.argv[2] if len(sys.argv) > 2 else "default_user"
    
    if not os.path.exists(pdf_path):
        print(f"❌ 文件不存在: {pdf_path}")
        sys.exit(1)
    
    print(f"📄 PDF: {pdf_path}")
    print(f"👤 User ID: {user_id}")
    print(f"=" * 60)
    
    # 延迟导入，减少初始内存占用
    from veragents.memory.rag.pipeline import RAGPipeline
    
    pipeline = RAGPipeline(
        knowledge_base_path=f"./data/kb_{user_id}",
        collection_name=f"qa_assistant_{user_id}",
        rag_namespace=f"pdf_{user_id}",
        chunk_tokens=512,
        overlap_tokens=64,
    )
    
    print(f"✅ RAG Pipeline 初始化完成 (dim={pipeline.dimension})")
    
    # 检查已有数据
    try:
        stats = pipeline.get_stats()
        vs = stats.get("vector_store", {})
        points = vs.get("points_count", 0)
        if points > 0:
            print(f"⚠️  已有 {points} 条向量数据")
            choice = input("是否清空后重新入库？(y/N): ").strip().lower()
            if choice == 'y':
                pipeline.clear()
                print("🗑️  已清空")
    except Exception:
        pass
    
    print(f"\n🚀 开始入库...")
    start = time.time()
    
    count = pipeline.ingest_file(pdf_path)
    
    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"✅ 入库完成！")
    print(f"   总分块: {count}")
    print(f"   耗时: {elapsed:.1f}s")
    print(f"\n💡 现在可以启动 Demo 进行问答:")
    print(f"   python demo/qa_assistant.py")
    print(f"   使用 User ID: {user_id}")

if __name__ == "__main__":
    main()
