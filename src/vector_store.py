import os
import faiss
import pandas as pd
from typing import List, Dict, Any
from .embedding import Embedder


class VectorStore:
    def __init__(self, index_path: str, meta_path: str, embedder: Embedder):
        self.index_path = index_path
        self.meta_path = meta_path
        self.embedder = embedder
        self.index = None
        self.meta = None

        # Ensure directories exist
        if os.path.dirname(index_path):
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
        if os.path.dirname(meta_path):
            os.makedirs(os.path.dirname(meta_path), exist_ok=True)

    def _ensure_loaded(self):
        """Lazy-load index and metadata if not already loaded"""
        if self.index is None and os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        if self.meta is None and os.path.exists(self.meta_path):
            self.meta = pd.read_parquet(self.meta_path)

    def build_from(self, papers: List[Dict[str, Any]]):
        """Build new FAISS index + metadata from scratch"""
        self.meta = pd.DataFrame(papers)
        texts = (self.meta["title"].fillna("") + "\n" + self.meta["abstract"].fillna("")).tolist()
        vecs = self.embedder.encode(texts)

        dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product similarity
        self.index.add(vecs)

        faiss.write_index(self.index, self.index_path)
        self.meta.to_parquet(self.meta_path, index=False)

    def add(self, papers: List[Dict[str, Any]]):
        """Add new papers to existing index + metadata"""
        self._ensure_loaded()
        new_df = pd.DataFrame(papers)

        if self.meta is None or self.index is None or len(getattr(self.meta, "columns", [])) == 0:
            self.build_from(papers)
            return

        self.meta = pd.concat([self.meta, new_df], ignore_index=True)
        texts = (new_df["title"].fillna("") + "\n" + new_df["abstract"].fillna("")).tolist()
        vecs = self.embedder.encode(texts)

        # Dimension check
        if self.index.d != vecs.shape[1]:
            raise ValueError(f"Embedding dimension mismatch: index {self.index.d}, new {vecs.shape[1]}")

        self.index.add(vecs)

        faiss.write_index(self.index, self.index_path)
        self.meta.to_parquet(self.meta_path, index=False)

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """Search for nearest papers"""
        self._ensure_loaded()

        if self.index is None or self.meta is None or len(self.meta) == 0:
            raise RuntimeError("Empty library. Add papers first.")

        q = self.embedder.encode([query])
        scores, idxs = self.index.search(q, top_k)

        idxs_1d = idxs.reshape(-1)
        scores_1d = scores.reshape(-1)

        valid = idxs_1d >= 0
        idxs_1d = idxs_1d[valid]
        scores_1d = scores_1d[valid]

        if idxs_1d.size == 0:
            return self.meta.head(0).copy()

        hits = self.meta.take(idxs_1d).copy()
        hits["score"] = scores_1d
        return hits.reset_index(drop=True)
