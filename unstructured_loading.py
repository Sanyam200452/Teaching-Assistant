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

print(docs[0].page_content)
print(docs[0].metadata)