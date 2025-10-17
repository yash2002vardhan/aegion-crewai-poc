# memory_with_metadata.py
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from openai import OpenAI
from datetime import datetime
import uuid
from typing import Optional, List, Dict
from dotenv import load_dotenv
import pandas as pd
load_dotenv(override=True)

class QdrantMemoryWithMetadata:
    def __init__(self, collection_name="customer_success_memory", qdrant_url=None, qdrant_api_key=None, embedding_model="text-embedding-3-large"):
        self.collection_name = collection_name
        qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY", None)
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.embedding_model = embedding_model
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # create collection if not exists (simple settings; tune as needed)
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest_models.VectorParams(size=3072, distance=rest_models.Distance.COSINE)  # 1536 typical; confirm with embedding model
            )

    def _embed(self, text: str) -> List[float]:
        # Using new OpenAI client (v1.0.0+)
        resp = self.openai_client.embeddings.create(model=self.embedding_model, input=text)
        return resp.data[0].embedding

    def add(self, text: str, metadata: dict):
        """
        metadata: must include at least sender_id, source, timestamp (optional, will be filled)
        """
        metadata = metadata.copy() if metadata else {}
        metadata.setdefault("id", str(uuid.uuid4()))
        metadata["text"] = text  # store original text in payload

        vector = self._embed(text)

        point = rest_models.PointStruct(
            id=metadata["id"],
            vector=vector,
            payload=metadata
        )
        self.client.upsert(collection_name=self.collection_name, points=[point])

        return metadata["id"]

    def retrieve(self, query_text: str, top_k: int = 5, user_id: Optional[str] = None, min_score: Optional[float] = None):
        """
        Returns list of dicts: [{id, payload, score}, ...]
        If user_id provided, applies filter to only return messages of that sender (using sender_id field).
        """
        vector = self._embed(query_text)

        # build filter - using sender_id field for consistency
        query_filter = None
        if user_id:
            query_filter = rest_models.Filter(must=[rest_models.FieldCondition(key="sender_id", match=rest_models.MatchValue(value=user_id))])

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False
        )

        results = []
        for r in search_result:
            # qdrant returns score in r.score depending on SDK; normalize if needed
            results.append({
                "id": r.id,
                "payload": r.payload,
                "score": getattr(r, "score", None)
            })

        # Optionally apply min_score filter
        if min_score is not None:
            results = [r for r in results if (r["score"] is not None and r["score"] >= min_score)]

        return results
    
    def retrieve_by_metadata(self, query_text: str, filters: Dict[str, str], top_k: int = 5, min_score: Optional[float] = None):
        """
        Retrieve similar vectors with metadata filtering.
        
        Args:
            query_text: Text to search for
            filters: Dictionary of metadata filters (e.g., {"role": "tool_success"})
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of dicts: [{id, payload, score}, ...]
        """
        vector = self._embed(query_text)
        
        # Build filter from provided filters dict
        filter_conditions = []
        for key, value in filters.items():
            filter_conditions.append(
                rest_models.FieldCondition(
                    key=key,
                    match=rest_models.MatchValue(value=value)
                )
            )
        
        query_filter = None
        if filter_conditions:
            query_filter = rest_models.Filter(must=filter_conditions)
        
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False
            )
            
            results = []
            for r in search_result:
                results.append({
                    "id": r.id,
                    "payload": r.payload,
                    "score": getattr(r, "score", None)
                })
            
            # Apply min_score filter if provided
            if min_score is not None:
                results = [r for r in results if (r["score"] is not None and r["score"] >= min_score)]
            
            return results
        except Exception as e:
            # If no results found or collection empty, return empty list
            return []
    
    def get_all_data(self):
        scroll_offset = None
        all_points = []

        while True:
            points, scroll_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=500,
                with_payload=True,
                with_vectors=False,
                offset=scroll_offset
            )
            all_points.extend(points)
            if scroll_offset is None:
                break

        return all_points


if __name__ == "__main__":
    df = pd.read_csv("/Users/vaibhavsatija/work/aegion-crewai-poc/-4886940973_20251014_164424.csv")

    qdrant_memory = QdrantMemoryWithMetadata(
        collection_name="customer_success_memory",
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        embedding_model="text-embedding-3-large"
    )
    for index, row in df.iterrows():
        metadata = {
            "original_channel_id":str(row["chat_id"]),
            "sender_id": str(row["sender_id"]),
            "source": 'telegram',
            "message_id": str(row["message_id"]),
            "role": "user",
            "timestamp": str(row["timestamp"])
        }
        qdrant_memory.add(text=str(row["message_content"]), metadata=metadata)
