from dotenv import load_dotenv
import os

from qdrant_client import QdrantClient

# Load variables from .env
load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

collection_name = "d2l-book"

collections = [c.name for c in client.get_collections().collections]
print(collections)
if collection_name in collections:
    
    client.delete_collection(collection_name)
    print(f"Deleted collection '{collection_name}'.")
else:
    print(f"Collection '{collection_name}' does not exist.")