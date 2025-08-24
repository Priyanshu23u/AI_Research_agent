import torch
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize: bool = True,
    ):
        # Pick device: CUDA > MPS (Apple) > CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"[Embedder] Loading model '{model_name}' on {self.device}...")

        # Safe load then move to device to avoid meta-tensor errors
        self.model = SentenceTransformer(model_name, device=None)
        try:
            self.model.to(self.device)
        except NotImplementedError:
            # Some SentenceTransformer submodules don't support .to() cleanly
            for _, module in self.model._modules.items():
                try:
                    module.to_empty(device=self.device)  # type: ignore[attr-defined]
                except Exception:
                    module.to(self.device)

        self.batch_size = batch_size
        self.normalize = normalize

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
            raise ValueError("Input must be a non-empty string or list of non-empty strings.")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            show_progress_bar=False,
            device=self.device,
        )

        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms

        return embeddings
