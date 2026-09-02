"""测试 Chroma 写入 - 用最简方式"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Test 1: Chroma in-memory (no disk)...", flush=True)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
    )

    test_docs = [
        Document(page_content="禁止买卖人体器官", metadata={"source": "test"}),
        Document(page_content="医师应当坚持人民至上", metadata={"source": "test2"}),
    ]

    # 先试不持久化（纯内存）
    vs = Chroma.from_documents(documents=test_docs, embedding=embeddings)
    results = vs.similarity_search("器官买卖", k=1)
    print(f"  In-memory OK! Found: {results[0].page_content}", flush=True)
except Exception as e:
    print(f"  In-memory FAILED: {e}", flush=True)

print("\nTest 2: Chroma with persist...", flush=True)
try:
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="chroma_test_")
    print(f"  Using temp dir: {tmp_dir}", flush=True)
    vs2 = Chroma.from_documents(
        documents=test_docs,
        embedding=embeddings,
        persist_directory=tmp_dir,
    )
    results2 = vs2.similarity_search("器官买卖", k=1)
    print(f"  Persist OK! Found: {results2[0].page_content}", flush=True)

    # 清理
    import shutil
    shutil.rmtree(tmp_dir)
except Exception as e:
    print(f"  Persist FAILED: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("\nDone!", flush=True)
