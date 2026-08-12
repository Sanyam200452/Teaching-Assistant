"""
chunker.py — Document loading and chunking using Docling.

Replaces unstructured.io entirely. Docling does its own PDF layout
parsing (not pdfminer + a graphics-op heuristic) and ships a
HybridChunker that chunks by document structure (headings, paragraphs,
tables, code blocks) while respecting a token budget.

Why the switch from unstructured:
  - unstructured's is_pdf_too_complex() heuristic silently returned 0
    elements on this book when strategy="fast" was forced — no error,
    just an empty list, because a few pages have dense vector graphics
    (matplotlib figures) that tripped a graphics-ops threshold meant
    for CAD drawings.
  - unstructured's classifier pipeline pulls in spaCy, which tries to
    auto-download a model from GitHub on first use — another silent
    failure point if that download is blocked.
  - Docling does real layout parsing itself and doesn't have either
    of these failure modes.

pip install docling langchain-docling
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from langchain_core.documents import Document


@dataclass
class Chunk:
    """Our internal chunk representation, used throughout the pipeline."""
    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0   # filled in during retrieval


class DocumentChunker:
    """
    Loads and chunks documents using Docling.

    Docling parses the PDF into a structured document (headings,
    paragraphs, tables, code, figures) and HybridChunker splits it
    into token-budgeted chunks that respect that structure — a chunk
    never spans two sections and code blocks aren't cut mid-way.
    """

    def __init__(
        self,
        max_tokens: int = 512,      # roughly maps to your old chunk_size=768 chars
        strategy: str = "fast",     # kept for interface compatibility, unused by Docling
    ):
        self.max_tokens = max_tokens
        self.strategy = strategy    # not used — Docling doesn't have a fast/hi_res split

    def load_and_chunk(self, path: Path) -> list[Chunk]:
        """Main entry point — load a file and return Chunks."""
        ext = path.suffix.lower()
        print(f"  📄 Loading {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)...")

        if ext not in {".pdf", ".docx", ".html", ".htm", ".md", ".txt"}:
            print(f"  ⚠️  Unsupported: {ext}")
            return []

        # DoclingLoader handles PDF, DOCX, HTML, MD, and more via the
        # same interface — Docling auto-detects the format.
        loader = DoclingLoader(
            file_path=str(path),
            export_type=ExportType.DOC_CHUNKS,
            chunker=HybridChunker(max_tokens=self.max_tokens),
        )

        docs = loader.load()
        print(f"     Raw chunks from Docling: {len(docs)}")

        return self._docs_to_chunks(docs, source=path.name)

    def chunk_directory(self, path: Path) -> list[Chunk]:
        """Chunk all supported files in a directory."""
        SUPPORTED = {".pdf", ".docx", ".txt", ".html", ".htm", ".md"}
        all_chunks: list[Chunk] = []
        for f in sorted(path.rglob("*")):
            if f.suffix.lower() in SUPPORTED:
                all_chunks.extend(self.load_and_chunk(f))
        return all_chunks

    # ------------------------------------------------------------------ #
    #  CONVERSION — Docling's chunk metadata → our Chunk dataclass        #
    # ------------------------------------------------------------------ #

    def _docs_to_chunks(self, docs: list[Document], source: str) -> list[Chunk]:
        """
        Docling's DOC_CHUNKS export gives each LangChain Document a
        `dl_meta` field in metadata — a dict describing where in the
        document this chunk came from: headings, page numbers, and the
        types of the original doc items (text, code, table, picture...).
        """
        result: list[Chunk] = []
        element_types = Counter()

        for doc in docs:
            text = doc.page_content.strip()
            if len(text) < 20:
                continue

            dl_meta = doc.metadata.get("dl_meta", {}) or {}

            chunk_meta = {"source": source}

            # Page number — Docling stores provenance per doc item;
            # take the page of the first item in this chunk.
            page = self._extract_page(dl_meta)
            if page:
                chunk_meta["page"] = page

            # Section — the nearest heading above this chunk, if any.
            headings = dl_meta.get("headings") or []
            if headings:
                chunk_meta["section"] = headings[-1]

            # Element type breakdown — used to flag code blocks etc.
            doc_items = dl_meta.get("doc_items") or []
            labels = [item.get("label", "text") for item in doc_items]
            primary_label = labels[0] if labels else "text"
            chunk_meta["element_type"] = primary_label
            chunk_meta["has_code"] = "code" in labels

            element_types[primary_label] += 1

            result.append(Chunk(text=text, metadata=chunk_meta))

        for label, count in element_types.most_common(6):
            print(f"       {label}: {count}")
        print(f"     → {len(result)} chunks from {source}")
        return result

    def _extract_page(self, dl_meta: dict) -> int | None:
        """Pull the page number from the first doc item's provenance, if present."""
        doc_items = dl_meta.get("doc_items") or []
        if not doc_items:
            return None
        prov = doc_items[0].get("prov") or []
        if not prov:
            return None
        return prov[0].get("page_no")
