# test_bm25_count.py
import pickle
with open("bm25_index.pkl", "rb") as f:
    data = pickle.load(f)
print("Chunks in bm25_index.pkl:", len(data["chunks"]))