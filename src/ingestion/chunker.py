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

from langchain_docling.loader import DoclingLoader
from langchain_docling.loader import ExportType
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.datamodel.base_models import InputFormat
import os 
from pathlib import Path
import json
from docling.datamodel.base_models import InputFormat
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

from dotenv import load_dotenv
import re

load_dotenv()
os.environ["TORCHDYNAMO_DISABLE"] = "1"


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


    def test_load_and_chunk(
        self,
        path: Path,
        page_range: tuple[int, int] | None = None,
    ) -> list:
        """
        Load and chunk a PDF with Docling.

        page_range:
            Optional (start_page, end_page).
            Example: (1, 20) processes only pages 1-20.
            Use None to process the entire PDF.
        """

        ext = path.suffix.lower()

        print(
            f"📄 Loading {path.name} "
            f"({path.stat().st_size / 1024 / 1024:.1f} MB)..."
        )

        if ext not in {".pdf", ".docx", ".html", ".htm", ".md", ".txt"}:
            print(f"⚠️ Unsupported: {ext}")
            return []

        # ----------------------------
        # Docling pipeline configuration
        # ----------------------------

        pipeline_options = PdfPipelineOptions()

        # D2L is a native-text PDF, so OCR isn't necessary.
        pipeline_options.do_ocr = False

        # If you don't need table extraction during this test,
        # disabling it can reduce processing time.
        pipeline_options.do_table_structure = True

        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=AcceleratorDevice.AUTO,
        )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        # ----------------------------
        # LangChain Docling loader
        # ----------------------------

        loader = DoclingLoader(
            file_path=str(path),
            converter=converter,
            export_type=ExportType.DOC_CHUNKS,
            chunker=HybridChunker(max_tokens=self.max_tokens),

            # This is passed directly to converter.convert(...)
            convert_kwargs={
                "page_range": page_range
            } if page_range else {},
        )

        # ----------------------------
        # Run Docling
        # ----------------------------

        print("⚙️ Processing PDF...")

        docs = loader.load()

        print(f"✅ Created {len(docs)} chunks.")

        # ----------------------------
        # Save chunks locally
        # ----------------------------

        output_dir = Path("docling_output")
        output_dir.mkdir(exist_ok=True)

        json_path = output_dir / f"{path.stem}_chunks.jsonl"
        txt_path = output_dir / f"{path.stem}_inspection.txt"

        with open(json_path, "w", encoding="utf-8") as f_json, \
            open(txt_path, "w", encoding="utf-8") as f_txt:

            for i, doc in enumerate(docs):

                dl_meta = doc.metadata.get("dl_meta", {}) or {}
                headings = dl_meta.get("headings") or []

                # Store only the useful information for inspection.
                record = {
                    "chunk_id": i,
                    "headings": headings,
                    "metadata": doc.metadata,
                    "text": doc.page_content,
                }

                # JSONL = one JSON object per line
                f_json.write(
                    json.dumps(record, ensure_ascii=False, default=str) + "\n"
                )

                # Human-readable file
                f_txt.write("=" * 80 + "\n")
                f_txt.write(f"CHUNK {i}\n")
                f_txt.write(f"HEADINGS: {headings}\n")
                f_txt.write("-" * 80 + "\n")
                f_txt.write(doc.page_content)
                f_txt.write("\n\n")

        print(f"💾 Saved chunks to: {json_path}")
        print(f"👀 Inspection file: {txt_path}")

        return docs
    

    def load_and_chunk(self, path: Path) -> list[Chunk]:
        """Main entry point — load a file and return Chunks."""
        ext = path.suffix.lower()
        print(f"  📄 Loading {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)...")

        if ext not in {".pdf", ".docx", ".html", ".htm", ".md", ".txt"}:
            print(f"  ⚠️  Unsupported: {ext}")
            return []


        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False   # D2L is a native-text PDF, OCR unnecessary
        pipeline_options.accelerator_options = AcceleratorOptions(
                                                                    num_threads=8,
                                                                    device=AcceleratorDevice.AUTO,
                                                                )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        loader = DoclingLoader(
            file_path=str(path),
            converter=converter,    # ← pass it in here
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
        result: list[Chunk] = []
        element_types = Counter()

        for doc in docs:
            text = doc.page_content.strip()
            if len(text) < 20:
                continue

            dl_meta = doc.metadata.get("dl_meta", {}) or {}

            chunk_meta = {"source": source}

            page = self._extract_page(dl_meta)
            if page:
                chunk_meta["page"] = page

            headings = dl_meta.get("headings") or []
            if headings:
                chunk_meta["section"] = headings[-1].strip()

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




    