import os
from dotenv import load_dotenv
load_dotenv()
from qdrant_client import QdrantClient

client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)
print(client.get_collections())