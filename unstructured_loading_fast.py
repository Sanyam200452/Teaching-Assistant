import unstructured.partition.pdf as _unstructured_pdf

def _never_too_complex(*args, **kwargs) -> bool:
    return False

_unstructured_pdf.is_pdf_too_complex = _never_too_complex

from langchain_unstructured import UnstructuredLoader

loader = UnstructuredLoader(
    "d2l-en.pdf",
    chunking_strategy="by_title",
    max_characters=1200,
    new_after_n_chars=1000,
    overlap=200,
)
docs = loader.load()
print(len(docs))