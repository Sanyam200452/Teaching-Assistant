from chunker import DocumentChunker
from pathlib import Path
import os 
from dotenv import load_dotenv

load_dotenv()
os.environ["TORCHDYNAMO_DISABLE"] = "1"

test_chunker = DocumentChunker() 

test_chunker.test_load_and_chunk(Path("d2l-en.pdf"))