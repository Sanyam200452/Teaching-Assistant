from langchain_docling.loader import ExportType, DoclingLoader

loader = DoclingLoader(
    file_path="./d2l-en.pdf",
    export_type=ExportType.DOC_CHUNKS,   # returns pre-chunked LangChain Documents
)

docs = loader.load()
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)