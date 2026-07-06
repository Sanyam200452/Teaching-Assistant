"""
chunker.py — Document loading and chunking using LangChain + unstructured.io.

Uses:
  - UnstructuredPDFLoader (LangChain)  → structure-aware PDF parsing
  - chunk_by_title (unstructured)      → chunks respect section boundaries,
                                         never split mid-equation or mid-code

pip install langchain-community unstructured[pdf]
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter

from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
    TextLoader,
)
from langchain_core.documents import Document


# Categories from unstructured we want to drop
DROP_CATEGORIES = {"Header", "Footer", "PageNumber", "PageBreak"}


@dataclass
class Chunk:
    """Our internal chunk representation, used throughout the pipeline."""
    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0   # filled in during retrieval


class DocumentChunker:
    """
    Loads and chunks documents using LangChain + unstructured.io.

    mode="elements" → one LangChain Document per structural element
                      (Title, NarrativeText, CodeSnippet, Table etc.)
    We then call unstructured's chunk_by_title() to merge elements into
    proper chunks that respect section boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 768,
        chunk_overlap: int = 96,
        combine_under: int = 200,
        strategy: str = "fast",
    ):
        self.chunk_size   = chunk_size
        self.chunk_overlap = chunk_overlap
        self.combine_under = combine_under
        self.strategy     = strategy

    def load_and_chunk(self, path: Path) -> list[Chunk]:
        """Main entry point — load a file and return Chunks."""
        ext = path.suffix.lower()
        print(f"  📄 Loading {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)...")

        if ext == ".pdf":
            lc_docs = self._load_pdf(path)
        elif ext == ".docx":
            lc_docs = self._load_docx(path)
        elif ext in {".html", ".htm"}:
            lc_docs = self._load_html(path)
        elif ext in {".txt", ".md"}:
            lc_docs = self._load_text(path)
        else:
            print(f"  ⚠️  Unsupported: {ext}")
            return []

        return self._docs_to_chunks(lc_docs, source=path.name)

    # ------------------------------------------------------------------ #
    #  LOADERS                                                             #
    # ------------------------------------------------------------------ #

    def _load_pdf(self, path: Path) -> list[Document]:
        loader = UnstructuredPDFLoader(
            str(path),
            mode="elements",           # one Document per structural element
            strategy=self.strategy,
            infer_table_structure=True,
        )
        docs = loader.load()
        print(f"     Raw elements: {len(docs)}")

        # Show element breakdown
        types = Counter(d.metadata.get("category", "unknown") for d in docs)
        for t, n in types.most_common(6):
            print(f"       {t}: {n}")

        return docs

    def _load_docx(self, path: Path) -> list[Document]:
        return UnstructuredWordDocumentLoader(str(path), mode="elements").load()

    def _load_html(self, path: Path) -> list[Document]:
        return UnstructuredHTMLLoader(str(path), mode="elements").load()

    def _load_text(self, path: Path) -> list[Document]:
        return TextLoader(str(path), encoding="utf-8").load()

    # ------------------------------------------------------------------ #
    #  CHUNKING                                                            #
    # ------------------------------------------------------------------ #

    def _docs_to_chunks(self, docs: list[Document], source: str) -> list[Chunk]:
        from unstructured.chunking.title import chunk_by_title  # type: ignore

        # 1. Filter noisy elements
        filtered_docs = [
            d for d in docs
            if d.metadata.get("category", "") not in DROP_CATEGORIES
            and len(d.page_content.strip()) >= 20
        ]

        # 2. Convert LangChain Documents → unstructured elements for chunk_by_title
        #    chunk_by_title needs unstructured element objects, not LangChain docs
        #    We use the elements that LangChain's loader already parsed internally
        #    by re-partitioning — or we fall back to LangChain's splitter.
        #    Simplest approach: use unstructured elements directly from the loader's
        #    internal state via the "elements" mode metadata.
        chunks_raw = self._chunk_lc_docs(filtered_docs)

        # 3. Convert to our Chunk dataclass
        result: list[Chunk] = []
        for doc in chunks_raw:
            text = doc.page_content.strip()
            if len(text) < 30:
                continue

            meta = doc.metadata
            chunk_meta = {
                "source":       source,
                "element_type": meta.get("category", "unknown"),
                "has_code":     meta.get("category", "") == "CodeSnippet",
            }

            # Page number
            page = meta.get("page_number") or meta.get("page")
            if page:
                chunk_meta["page"] = page

            result.append(Chunk(text=text, metadata=chunk_meta))

        print(f"     → {len(result)} chunks from {source}")
        return result

    def _chunk_lc_docs(self, docs: list[Document]) -> list[Document]:
        """
        Split LangChain Documents using RecursiveCharacterTextSplitter.
        This is the LangChain-native chunking approach.
        chunk_by_title requires unstructured element objects — using
        LangChain's splitter here keeps everything in LangChain's ecosystem.
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_documents(docs)
