"""
Qdrant vector database client for managing embeddings.
Handles collection creation, upsert, and search operations.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from typing import List, Dict, Optional, Union
import uuid


class QdrantVectorStore:
    """Manages Qdrant vector database operations."""

    def __init__(
        self,
        collection_name: str = "rag_chunks",
        path: str = "./qdrant_local",
        vector_size: int = 512,
        distance: Distance = Distance.COSINE,
        recreate: bool = False
    ):
        """
        Initialize Qdrant client.

        Args:
            collection_name: Name of the collection.
            path: Path for local persistent storage.
            vector_size: Dimension of vectors (default 512).
            distance: Distance metric (COSINE, EUCLIDEAN, DOT).
            recreate: Whether to recreate collection if exists.
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance

        print(f"🧠 Initializing Qdrant client at: {path}")
        self.client = QdrantClient(path=path)
        self._ensure_collection(recreate)

    def _ensure_collection(self, recreate: bool = False):
        """Ensure collection exists, recreate if needed."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if recreate and exists:
            print(f"🔁 Recreating collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            print(f"🆕 Creating collection: {self.collection_name} (dim={self.vector_size})")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=self.distance)
            )
        else:
            info = self.client.get_collection(self.collection_name)
            current_size = info.config.params.vectors.size
            if current_size != self.vector_size:
                print(f"⚠️ Dimension mismatch ({current_size} vs {self.vector_size}). Recreating...")
                self.client.delete_collection(self.collection_name)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=self.distance)
                )

    def upsert(
        self,
        embeddings: List[List[float]],
        payloads: List[Dict],
        ids: Optional[List[Union[str, int]]] = None,
        batch_size: int = 100
    ):
        """
        Insert or update vectors in batches.
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in embeddings]

        for i in range(0, len(embeddings), batch_size):
            batch_emb = embeddings[i:i + batch_size]
            batch_payload = payloads[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            # Ensure proper dimensionality (truncate/pad)
            fixed_batch_emb = [
                emb[:self.vector_size] if len(emb) >= self.vector_size
                else emb + [0.0] * (self.vector_size - len(emb))
                for emb in batch_emb
            ]

            points = [
                PointStruct(id=id_val, vector=vec, payload=meta)
                for id_val, vec, meta in zip(batch_ids, fixed_batch_emb, batch_payload)
            ]

            self.client.upsert(collection_name=self.collection_name, points=points)

        print(f"✅ Upserted {len(embeddings)} points into {self.collection_name} (dim={self.vector_size})")

    def search(
        self,
        query_vector: List[float],
        limit: int = 3,
        score_threshold: Optional[float] = None,
        filter: Optional[Filter] = None
    ) -> List[Dict]:
        """
        Retrieve nearest vectors.
        """
        # Enforce query vector length 
        if len(query_vector) != self.vector_size:
            query_vector = (
                query_vector[:self.vector_size]
                if len(query_vector) > self.vector_size
                else query_vector + [0.0] * (self.vector_size - len(query_vector))
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=filter
        )

        return [
            {"id": r.id, "score": r.score, "payload": r.payload}
            for r in results
        ]

    def delete_by_filter(self, filter: Filter):
        """Delete points matching filter."""
        self.client.delete(collection_name=self.collection_name, points_selector=filter)

    def clear_collection(self):
        """Remove and recreate the collection."""
        self.client.delete_collection(self.collection_name)
        self._ensure_collection(recreate=False)

    def delete_by_file_id(self, file_id: str):
        """Delete all vectors associated with a given file ID."""
        filter_condition = Filter(
            must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))]
        )
        self.delete_by_filter(filter_condition)
