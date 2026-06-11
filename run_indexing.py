"""
run_indexing.py — index d2l-en.pdf into Qdrant Cloud once.

Usage:
    pip install -r requirements.txt
    # fill in .env file (see .env.example)
    python run_indexing.py --pdf ./d2l-en.pdf

Uses unstructured.io for document-aware chunking:
  - Preserves code blocks and equations intact
  - Chunks at section/title boundaries
  - Attaches richer metadata (page, section title, element type)
"""

import argparse
import sys
import time
from pathlib import Path
from dotenv import load_dotenv  # reads .env into os.environ

load_dotenv()

import os
for k in ("OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"):
    if not os.environ.get(k):
        print(f"❌ Missing env var: {k}")
        print(f"   Add it to your .env file — see .env.example")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="./d2l-en.pdf", help="Path to d2l-en.pdf")
    parser.add_argument("--strategy", default="fast", choices=["fast", "hi_res"],
                        help="fast = pdfminer (recommended). hi_res = OCR layout detection (slow).")
    parser.add_argument("--force", action="store_true", help="Re-index even if data exists")
    args = parser.parse_args()

    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"❌ PDF not found: {pdf}")
        print("   Download from: https://d2l.ai/d2l-en.pdf")
        sys.exit(1)

    from src.ingestion.chunker import UnstructuredChunker
    from src.ingestion.embedder import Embedder
    from src.retrieval.vector_store import VectorStore
    from src.retrieval.bm25_store import BM25Store

    # ── Check if already indexed ──────────────────────────────────────
    vs = VectorStore("d2l-book")
    if vs.collection_exists_and_has_data() and not args.force:
        print("✅ Collection already has data. Use --force to re-index.")
        sys.exit(0)

    # ── Chunk ─────────────────────────────────────────────────────────
    print(f"\n📖 Chunking {pdf.name} with unstructured.io (strategy={args.strategy})...")
    print("   This preserves code blocks, equations, and section boundaries.\n")

    chunker = UnstructuredChunker(
        max_characters=768,
        overlap=96,
        combine_under=200,
        strategy=args.strategy,
    )
    chunks = chunker.chunk_file(pdf)
    print(f"\n✅ {len(chunks)} chunks created")

    # Show element type breakdown
    from collections import Counter
    types = Counter(c.metadata.get("element_type", "unknown") for c in chunks)
    for t, n in types.most_common():
        print(f"   {t}: {n}")

    # ── Embed ─────────────────────────────────────────────────────────
    embedder = Embedder()
    print(f"\n🔢 Embedding {len(chunks)} chunks (~5 min for full book)...")

    all_embeddings = []
    batch = 100
    for i in range(0, len(chunks), batch):
        embs = embedder.embed([c.text for c in chunks[i:i + batch]])
        all_embeddings.extend(embs)
        pct = min(100, int((i + batch) / len(chunks) * 100))
        print(f"   {pct}%", end="\r")
        time.sleep(0.05)

    # ── Upload to Qdrant ──────────────────────────────────────────────
    print(f"\n📤 Uploading to Qdrant Cloud (collection: d2l-book)...")
    vs.upsert(chunks, all_embeddings)

    # ── Build BM25 index ──────────────────────────────────────────────
    print("📝 Building BM25 index...")
    BM25Store().index(chunks)

    print(f"\n✅ Done — {len(chunks)} chunks indexed.")
    print("   Commit bm25_index.pkl then deploy to Streamlit Cloud.")
    print("\n   Chunk metadata now includes:")
    has_page    = sum(1 for c in chunks if c.metadata.get("page"))
    has_section = sum(1 for c in chunks if c.metadata.get("section"))
    has_code    = sum(1 for c in chunks if c.metadata.get("has_code"))
    print(f"   - page number:    {has_page}/{len(chunks)} chunks")
    print(f"   - section title:  {has_section}/{len(chunks)} chunks")
    print(f"   - contains code:  {has_code}/{len(chunks)} chunks")


if __name__ == "__main__":
    main()
