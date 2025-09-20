import faiss
import numpy as np
import os
import pickle
from src.config import EMBEDDINGS_DIR

INDEX_FILE = os.path.join(EMBEDDINGS_DIR, "faiss.index")
META_FILE = os.path.join(EMBEDDINGS_DIR, "meta.pkl")

class VectorStore:
    def __init__(self, dim=1536):
        self.index = faiss.IndexFlatL2(dim)
        self.meta = []

    def add(self, embeddings, metadata):
        """Add vectors and metadata."""
        self.index.add(np.array(embeddings).astype("float32"))
        self.meta.extend(metadata)

    def query(self, vector, top_k=5):
        
        if self.index.ntotal == 0:
            return []
        
        D, I = self.index.search(np.array([vector]).astype("float32"), top_k)
        results = []

        for j, i in enumerate(I[0]):
            if 0 <= i < len(self.meta):
                results.append((self.meta[i], float(D[0][j])))
        
        return results

    def save(self):
        faiss.write_index(self.index, INDEX_FILE)

        with open(META_FILE, "wb") as f:
            pickle.dump(self.meta, f)

    def load(self):

        if os.path.exists(INDEX_FILE):
            self.index = faiss.read_index(INDEX_FILE)
            
        if os.path.exists(META_FILE):
            with open(META_FILE, "rb") as f:
                self.meta = pickle.load(f)
