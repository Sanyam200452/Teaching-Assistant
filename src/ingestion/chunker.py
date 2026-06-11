"""
chunker.py — Document-aware chunking using unstructured.io.

Replaces the old recursive character splitter. unstructured.io parses the
document structure FIRST (titles, narrative text, code blocks, tables,
formulas, list items) and THEN chunks — so it never splits mid-equation
or mid-code-block.

Strategy: chunk_by_title()
  - Starts a new chunk at every Title/Header element
  - Keeps code blocks, formulas, and list items intact
  - Merges short elements up to max_characters
  - Falls back to character splitting only when a single element exceeds max_characters

Two strategies for PDF parsing:
  "fast"    — pdfminer text extraction, no OCR. Fast, good for D2L (text-based PDF).
  "hi_res"  — layout detection + OCR via detectron2. Slow but handles scanned PDFs.
  Use "fast" for D2L — it's a text PDF and fast is plenty.

pip install "unstructured[pdf]"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0    # populated during retrieval


# Element types we care about — everything else (headers, footers, page nums) is dropped
KEEP_TYPES = {
    "NarrativeText",
    "Title",
    "ListItem",
    "CodeSnippet",
    "Table",
    "FigureCaption",
    "Formula",
    "EmailAddress",   # occasionally appears in academic PDFs
}


class UnstructuredChunker:
    """
    Document-aware chunker using unstructured.io.

    Parses structure first, then chunks by title boundaries.
    Preserves code blocks, equations, and list items intact.
    Attaches richer metadata: element_type, section title, page number.
    """

    def __init__(
        self,
        max_characters: int = 768,
        overlap: int = 96,
        combine_under: int = 200,   # merge elements shorter than this into neighbours
        strategy: str = "fast",     # "fast" or "hi_res"
    ):
        self.max_characters = max_characters
        self.overlap = overlap
        self.combine_under = combine_under
        self.strategy = strategy

    def chunk_file(self, path: Path) -> list[Chunk]:
        """
        Parse and chunk a single file.
        Main entry point — replaces the old loader.load() + chunker.chunk() pattern.
        """
        ext = path.suffix.lower()

        if ext == ".pdf":
            return self._chunk_pdf(path)
        elif ext == ".docx":
            return self._chunk_docx(path)
        elif ext in {".html", ".htm"}:
            return self._chunk_html(path)
        elif ext in {".txt", ".md"}:
            return self._chunk_text(path)
        else:
            print(f"  ⚠️  Unsupported file type: {ext}, skipping {path.name}")
            return []

    def chunk_directory(self, path: Path) -> list[Chunk]:
        """Chunk all supported files in a directory."""
        SUPPORTED = {".pdf", ".docx", ".txt", ".html", ".htm", ".md"}
        all_chunks: list[Chunk] = []
        for f in sorted(path.rglob("*")):
            if f.suffix.lower() in SUPPORTED:
                print(f"  📄 {f.name}")
                all_chunks.extend(self.chunk_file(f))
        return all_chunks

    # ------------------------------------------------------------------ #
    #  PARSERS                                                             #
    # ------------------------------------------------------------------ #

    def _chunk_pdf(self, path: Path) -> list[Chunk]:
        from unstructured.partition.pdf import partition_pdf  # type: ignore
        from unstructured.chunking.title import chunk_by_title  # type: ignore

        print(f"    Partitioning PDF (strategy={self.strategy})...")
        elements = partition_pdf(
            filename=str(path),
            strategy=self.strategy,
            # Extract page numbers into metadata
            include_page_breaks=False,
            # infer_table_structure gives you markdown tables — great for D2L
            infer_table_structure=True,
        )
        return self._to_chunks(elements, source=path.name)

    def _chunk_docx(self, path: Path) -> list[Chunk]:
        from unstructured.partition.docx import partition_docx  # type: ignore
        elements = partition_docx(filename=str(path))
        return self._to_chunks(elements, source=path.name)

    def _chunk_html(self, path: Path) -> list[Chunk]:
        from unstructured.partition.html import partition_html  # type: ignore
        elements = partition_html(filename=str(path))
        return self._to_chunks(elements, source=path.name)

    def _chunk_text(self, path: Path) -> list[Chunk]:
        from unstructured.partition.text import partition_text  # type: ignore
        elements = partition_text(filename=str(path))
        return self._to_chunks(elements, source=path.name)

    # ------------------------------------------------------------------ #
    #  CORE: elements → chunks                                            #
    # ------------------------------------------------------------------ #

    def _to_chunks(self, elements: list, source: str) -> list[Chunk]:
        from unstructured.chunking.title import chunk_by_title  # type: ignore

        # Filter noisy elements (page headers, footers, isolated page numbers)
        filtered = [e for e in elements if type(e).__name__ in KEEP_TYPES]

        chunks_raw = chunk_by_title(
            filtered,
            max_characters=self.max_characters,
            overlap=self.overlap,
            combine_text_under_n_chars=self.combine_under,
            # Don't split a code block even if it exceeds max_characters
            # (better to have one big code chunk than two broken halves)
            new_after_n_chars=self.max_characters + 200,
        )

        result: list[Chunk] = []
        for chunk in chunks_raw:
            text = chunk.text.strip()
            if len(text) < 30:
                continue

            # Rich metadata from unstructured
            meta = chunk.metadata
            chunk_meta = {
                "source": source,
                "element_type": type(chunk).__name__,
            }

            # Page number (not always available depending on strategy)
            if hasattr(meta, "page_number") and meta.page_number:
                chunk_meta["page"] = meta.page_number

            # Section title — the Title element that started this chunk
            if hasattr(meta, "section") and meta.section:
                chunk_meta["section"] = meta.section

            # For code blocks: tag them so we can display them differently in UI
            if type(chunk).__name__ == "CompositeElement":
                # Check if chunk contains a code snippet
                orig_types = {type(e).__name__ for e in getattr(chunk, "metadata", {}).get("orig_elements", [])}
                if "CodeSnippet" in orig_types:
                    chunk_meta["has_code"] = True

            result.append(Chunk(text=text, metadata=chunk_meta))

        print(f"    → {len(result)} chunks from {source}")
        return result
