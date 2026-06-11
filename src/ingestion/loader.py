"""
loader.py — thin compatibility shim.
With unstructured.io, parsing and chunking happen in one pass inside
UnstructuredChunker.chunk_file(). This shim keeps the RawDocument type
alive in case anything imports it, but new code should use
UnstructuredChunker directly.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RawDocument:
    """Kept for backwards compatibility. Not used by UnstructuredChunker."""
    text: str
    metadata: dict = field(default_factory=dict)
