import os
import pandas as pd
from typing import List, Dict, Any
from .embedding import Embedder
from .vector_store import VectorStore

LIB_DIR = os.path.join("data", "library")
INDEX_PATH = os.path.join(LIB_DIR, "index.faiss")
META_PATH = os.path.join(LIB_DIR, "meta.parquet")


class Library:
    def __init__(self, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        os.makedirs(LIB_DIR, exist_ok=True)
        self.embedder = Embedder(embed_model)
        self.store = VectorStore(INDEX_PATH, META_PATH, self.embedder)

        # Ensure metadata parquet exists
        if not os.path.exists(META_PATH):
            pd.DataFrame([]).to_parquet(META_PATH)

    def _normalize_paper(self, p: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize input paper dict into a consistent schema."""
        authors = p.get("authors", "")
        if isinstance(authors, list):
            authors = ", ".join([str(a).strip() for a in authors if str(a).strip()])

        categories = p.get("categories", "")
        if isinstance(categories, list):
            categories = ", ".join([str(c).strip() for c in categories if str(c).strip()])

        return {
            "id": p.get("id") or p.get("link_abs") or "",
            "title": p.get("title", ""),
            "abstract": p.get("abstract", p.get("summary", "")),
            "authors": authors,
            "published": p.get("published", ""),
            "year": str(p.get("year", "") or "").strip(),
            "link_abs": p.get("link_abs") or p.get("link") or "",
            "link_pdf": p.get("link_pdf", ""),
            "categories": categories,
        }

    def add_papers(self, papers: List[Dict[str, Any]]) -> int:
        """Add normalized papers to the vector store."""
        if not papers:
            return 0
        normed = [self._normalize_paper(p) for p in papers]
        self.store.add(normed)
        return len(normed)

    def list(self) -> pd.DataFrame:
        """List all stored papers with metadata."""
        self.store._ensure_loaded()
        return self.store.meta.copy() if self.store.meta is not None else pd.DataFrame([])

    def get_by_ids(self, ids: List[int]) -> pd.DataFrame:
        """Fetch papers by row indices."""
        meta = self.list()
        if meta.empty:
            return meta
        ids_valid = [i for i in ids if 0 <= i < len(meta)]
        if not ids_valid:
            return pd.DataFrame([])
        return meta.iloc[ids_valid].copy()

    def retrieve(self, query: str, top_k: int = 12) -> pd.DataFrame:
        """Retrieve top_k most relevant papers for a query."""
        return self.store.search(query, top_k=top_k)

    def clear(self):
        """Clear the entire library (index + metadata)."""
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
        if os.path.exists(META_PATH):
            os.remove(META_PATH)
        os.makedirs(LIB_DIR, exist_ok=True)
        pd.DataFrame([]).to_parquet(META_PATH)
