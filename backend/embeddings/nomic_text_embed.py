"""
Text embeddings using nomic-embed-text-v1.5 model.
Generates dense embeddings with Matryoshka dimension reduction (512D default).
"""

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional
import numpy as np
from tqdm import tqdm


class NomicTextEmbedder:
    """Generate text embeddings using Nomic Embed Text model."""

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        matryoshka_dim: int = 512,
        batch_size: int = 64,
        device: Optional[str] = None
    ):
        """
        Initialize Nomic text embedder.

        Args:
            model_name: HuggingFace model identifier.
            matryoshka_dim: Dimension for Matryoshka reduction (default 512).
            batch_size: Batch size for processing.
            device: Device to use ('cuda', 'cpu', or None for auto).
        """
        self.model_name = model_name
        self.matryoshka_dim = matryoshka_dim
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"📦 Loading Nomic text embedding model: {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=self.device)
        self.model.eval()

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = True,
        convert_to_numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode texts into normalized embeddings (512D).

        Args:
            texts: Single text string or list of texts.
            show_progress: Show progress bar during encoding.
            convert_to_numpy: Convert to NumPy array instead of Torch tensor.

        Returns:
            Embeddings tensor or array.
        """
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        for i in tqdm(range(0, len(texts), self.batch_size),
                      desc="Encoding texts", disable=not show_progress):
            batch = texts[i:i + self.batch_size]
            batch_prefixed = [f"search_query: {text}" for text in batch]

            with torch.no_grad():
                embeddings = self.model.encode(batch_prefixed, convert_to_tensor=True)

            # Normalize per layer
            embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))

            # Matryoshka dimension reduction (safe slice/pad)
            if embeddings.shape[1] >= self.matryoshka_dim:
                embeddings = embeddings[:, :self.matryoshka_dim]
            else:
                pad_size = self.matryoshka_dim - embeddings.shape[1]
                embeddings = F.pad(embeddings, (0, pad_size))

            # L2 normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())

        result = torch.cat(all_embeddings, dim=0)
        return result.numpy() if convert_to_numpy else result

    def encode_with_metadata(
        self,
        texts: List[str],
        metadata: Optional[List[dict]] = None,
        show_progress: bool = True
    ) -> List[dict]:
        """
        Encode texts and return with metadata.
        """
        embeddings = self.encode(texts, show_progress=show_progress, convert_to_numpy=True)
        results = []
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            record = {"embedding": emb.tolist(), "text": text}
            if metadata and i < len(metadata):
                record.update(metadata[i])
            results.append(record)
        return results
