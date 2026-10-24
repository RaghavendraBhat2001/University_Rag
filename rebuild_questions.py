# rebuild_questions.py
"""
Rebuilds the Question Bank and Question Chunks using the upgraded high-precision extractor.
Preserves existing notes FAISS indexes and completes in seconds.
"""
import sys
from pathlib import Path
import json
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

root = Path(__file__).resolve().parent  # REVIEW_FIXES_APPLIED (was a hard-coded personal path)
sys.path.insert(0, str(root))

from config import FAISS_DIR, PAPERS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.document_loader import load_pdf
from src.question_index import build_question_bank, save_question_bank, build_question_vectorstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

def rebuild():
    print("📋 Rebuilding Question Bank with high-precision question extractor...")
    if not PAPERS_DIR.exists():
        print("❌ PAPERS_DIR does not exist!")
        return

    qp_pages = []
    for pdf in sorted(PAPERS_DIR.glob("**/*.pdf")):
        docs = load_pdf(pdf, "question_paper")
        qp_pages.extend(docs)

    print(f"📄 Loaded {len(qp_pages)} question paper pages across all subjects.")

    # 1. Build structured question bank
    question_bank = build_question_bank(qp_pages)
    print(f"✅ Extracted {len(question_bank)} individual question records.")
    
    qb_path = FAISS_DIR / "_question_bank.json"
    save_question_bank(question_bank, qb_path)
    print(f"💾 Saved to {qb_path}")

    # 2. Save plain text chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    paper_chunks = splitter.split_documents(qp_pages)
    papers_path = FAISS_DIR / "_papers.json"
    with open(papers_path, "w", encoding="utf-8") as f:
        json.dump([{"page_content": c.page_content, "metadata": c.metadata} for c in paper_chunks], f, ensure_ascii=False)
    print(f"💾 Saved {len(paper_chunks)} question paper chunks to {papers_path}")

    # 3. Build Question Vectorstore for semantic question matching
    print("💾 Embedding Question Bank for semantic question retrieval...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://127.0.0.1:11434")
    qb_store = build_question_vectorstore(question_bank, embeddings, store_path=FAISS_DIR / "_question_bank_faiss")
    if qb_store:
        print("✅ Question vector store built and saved to _question_bank_faiss")

    # Update manifest
    manifest_path = FAISS_DIR / "_manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            pass
    manifest["status"] = "complete"
    manifest["question_bank_size"] = len(question_bank)
    manifest["paper_chunks"] = len(paper_chunks)
    manifest["question_vectorstore_built"] = qb_store is not None
    manifest["updated_at"] = datetime.now().isoformat()
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("🎉 Question bank rebuild complete!")

if __name__ == "__main__":
    rebuild()
