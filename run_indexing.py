"""
run_indexing.py — Index d2l-en.pdf into Qdrant Cloud once.

Usage:
    pip install -r requirements.txt
    # fill in .env (copy from .env.example)
    python run_indexing.py --pdf ./d2l-en.pdf

After running:
    git add bm25_index.pkl
    git commit -m "add BM25 index"
    # then deploy to Streamlit Cloud
"""

import argparse
import os
import sys
import time
import pickle
from pathlib import Path
from dotenv import load_dotenv
import json

load_dotenv()
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Validate env vars before doing anything
REQUIRED = ["OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "COHERE_API_KEY"]
for key in REQUIRED:
    if not os.environ.get(key):
        print(f"❌ Missing: {key}  →  add it to your .env file")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf",        default="./d2l-en.pdf")
    parser.add_argument("--max-tokens", type=int, default=512,
                         help="Token budget per chunk for Docling's HybridChunker "
                              "(roughly maps to the old chunk_size in characters)")
    parser.add_argument("--force", action="store_true", help="Re-index even if data exists")
    args = parser.parse_args()

    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"❌ PDF not found: {pdf}")
        print("   Download: https://d2l.ai/d2l-en.pdf")
        sys.exit(1)

    from src.ingestion.chunker import DocumentChunker
    from src.ingestion.embedder import Embedder
    from src.retrieval.vector_store import VectorStore
    from src.retrieval.bm25_store import BM25Store

    # ── Check if already indexed ──────────────────────────────────────
    vs = VectorStore("d2l-book")
    if vs.collection_exists_and_has_data() and not args.force:
        print("✅ Already indexed. Use --force to re-index.")
        sys.exit(0)

    # ── Load + chunk ──────────────────────────────────────────────────
    # Docling does its own layout parsing (no fast/hi_res split like
    # unstructured had) and HybridChunker chunks by document structure
    # within a token budget rather than a character count.
    print(f"\n📖 Loading and chunking {pdf.name} (max_tokens={args.max_tokens})...")
    print("   First run downloads Docling's layout model (~a few hundred MB, cached after).")

    chunker = DocumentChunker(max_tokens=args.max_tokens)


    cache_path = Path("chunks_cache.pkl")
    if cache_path.exists() and not args.force:
        print("📦 Loading chunks from cache...")
        with open(cache_path, "rb") as f:
            chunks = pickle.load(f)
    else:
        chunks = chunker.load_and_chunk(pdf)
        with open(cache_path, "wb") as f:
            pickle.dump(chunks, f)
    print("   (cached to chunks_cache.pkl in case embedding fails)")
    print(f"\n✅ {len(chunks)} chunks created")

    if not chunks:
        print("❌ No chunks produced. Check the PDF.")
        sys.exit(1)

    chapter_pairs = sorted(
    {
        (c.metadata["chapter"], c.metadata.get("chapter_title", ""))
        for c in chunks
        if c.metadata.get("chapter")
    },
    key=lambda pair: int(pair[0]),
)
 
    with open("chapters.json", "w", encoding="utf-8") as f:
        json.dump(chapter_pairs, f, indent=2)
    
    print(f"   📚 {len(chapter_pairs)} chapters found → saved to chapters.json")
    for num, title in chapter_pairs:
        print(f"      {num}. {title}")

    # ── Embed ─────────────────────────────────────────────────────────
    embedder = Embedder()
    print(f"\n🔢 Embedding {len(chunks)} chunks...")

    all_embeddings = []
    batch = 100
    for i in range(0, len(chunks), batch):
        embs = embedder.embed([c.text for c in chunks[i : i + batch]])
        all_embeddings.extend(embs)
        pct = min(100, int((i + batch) / len(chunks) * 100))
        print(f"   {pct}%", end="\r")
        time.sleep(0.05)

    # ── Upload to Qdrant ──────────────────────────────────────────────
    print(f"\n📤 Uploading to Qdrant Cloud...")
    vs.upsert(chunks, all_embeddings)

    # ── Build BM25 index ──────────────────────────────────────────────
    print("📝 Building BM25 index...")
    BM25Store().index(chunks)

    # ── Summary ───────────────────────────────────────────────────────
    has_page    = sum(1 for c in chunks if c.metadata.get("page"))
    has_section = sum(1 for c in chunks if c.metadata.get("section"))
    has_code    = sum(1 for c in chunks if c.metadata.get("has_code"))
    print(f"\n✅ Done — {len(chunks)} chunks indexed.")
    print(f"   Page numbers:   {has_page}/{len(chunks)} chunks")
    print(f"   Section titles: {has_section}/{len(chunks)} chunks")
    print(f"   Code chunks:    {has_code}/{len(chunks)} chunks")
    print("\n   Next steps:")
    print("   1. git add bm25_index.pkl && git commit -m 'add BM25 index'")
    print("   2. Deploy to Streamlit Cloud")


if __name__ == "__main__":
    main()