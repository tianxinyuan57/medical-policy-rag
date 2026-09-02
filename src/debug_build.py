"""逐步调试建库流程"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Step 0: Starting...", flush=True)

try:
    print("Step 1: Loading documents...", flush=True)
    from build_index import load_documents, split_documents
    docs = load_documents()
    print(f"  Loaded {len(docs)} docs", flush=True)

    print("Step 2: Splitting...", flush=True)
    chunks = split_documents(docs)
    print(f"  Split into {len(chunks)} chunks", flush=True)

    print("Step 3: Loading embedding model...", flush=True)
    from langchain_huggingface import HuggingFaceEmbeddings
    from config import EMBED_MODEL, CHROMA_DIR
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("  Embedding model loaded!", flush=True)

    print("Step 4: Creating Chroma vectorstore...", flush=True)
    from langchain_chroma import Chroma

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chroma_path = os.path.join(project_root, CHROMA_DIR)

    # 确保目录干净
    import shutil
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
        print("  Cleared old chroma_db", flush=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_path,
    )
    print("  Vectorstore created!", flush=True)

    print("Step 5: Test search...", flush=True)
    results = vectorstore.similarity_search("器官捐献年龄限制", k=3)
    for i, doc in enumerate(results):
        source = os.path.basename(doc.metadata.get("source", "?"))
        print(f"  [{i+1}] {source}: {doc.page_content[:60]}...", flush=True)

    print("\nAll done!", flush=True)

except Exception as e:
    print(f"\nERROR at current step: {e}", flush=True)
    import traceback
    traceback.print_exc()
