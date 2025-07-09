"""ChromaDB vector store implementation for REM RAG"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime
import uuid
import random

from ..config import (
    CHROMA_PERSIST_DIR, 
    COLLECTION_NAME, 
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS
)

logger = logging.getLogger(__name__)


class REMVectorStore:
    """
    Vector store for REM RAG using ChromaDB.
    
    Handles storage and retrieval of knowledge nodes with metadata tracking
    for years, entities, and generation depth.
    """
    
    def __init__(self, collection_name: str = COLLECTION_NAME):
        """Initialize ChromaDB with OpenAI embeddings."""
        self.collection_name = collection_name
        
        # Ensure persist directory exists
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_PERSIST_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Set up OpenAI embedding function
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=EMBEDDING_MODEL
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception as e:
            # Handle both ValueError and NotFoundError
            logger.info(f"Collection not found, creating new: {collection_name}")
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add(self, texts: List[str], metadata: List[dict]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text documents to embed and store
            metadata: List of metadata dicts with fields like:
                - year: int (publication year)
                - entity: str (associated entity/concept)
                - source_type: str ("article", "synthesis", "rem_dream")
                - generation_depth: int (0 for original, 1+ for syntheses)
                - created_at: str (ISO timestamp)
                - article_id: str (original article ID if applicable)
                
        Returns:
            List of generated document IDs
        """
        if len(texts) != len(metadata):
            raise ValueError("Texts and metadata must have same length")
        
        # Generate IDs
        ids = [str(uuid.uuid4()) for _ in texts]
        
        # Add creation timestamp if not present
        for meta in metadata:
            if 'created_at' not in meta:
                meta['created_at'] = datetime.now().isoformat()
        
        # Add to collection
        self.collection.add(
            documents=texts,
            metadatas=metadata,
            ids=ids
        )
        
        logger.info(f"Added {len(texts)} documents to collection")
        return ids
    
    def query(self, text: str, k: int = 5, filter: Optional[dict] = None) -> dict:
        """
        Similarity search with optional filtering.
        
        Args:
            text: Query text
            k: Number of results to return
            filter: ChromaDB where clause, e.g.:
                {"year": {"$gte": 1960, "$lte": 1965}}
                {"entity": "Soviet Union"}
                {"source_type": "synthesis"}
                
        Returns:
            Dict with keys: documents, metadatas, distances, ids
        """
        query_params = {
            "query_texts": [text],
            "n_results": k
        }
        
        if filter:
            query_params["where"] = filter
        
        results = self.collection.query(**query_params)
        
        # Flatten results since we only have one query
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "ids": results["ids"][0] if results["ids"] else []
        }
    
    def sample(self, n: int, filter: Optional[dict] = None) -> dict:
        """
        Random sampling for REM cycles.
        
        Args:
            n: Number of documents to sample
            filter: Optional ChromaDB where clause
            
        Returns:
            Dict with keys: documents, metadatas, ids
        """
        # Use offset-based random sampling for true uniform distribution
        
        # First, get the total count
        # Since ChromaDB count() doesn't support filters, we get a minimal query
        if filter:
            # Get one result to check if any documents match
            test_result = self.collection.get(where=filter, limit=1, include=[])
            if not test_result["ids"]:
                return {"documents": [], "metadatas": [], "ids": []}
            
            # For filtered queries, we need to get all IDs first
            # This is a limitation of ChromaDB - no filtered count
            all_ids = self.collection.get(
                where=filter, 
                limit=100000,  # High limit
                include=[]  # Just IDs for efficiency
            )
            total_count = len(all_ids["ids"])
        else:
            # For unfiltered, we can use count()
            total_count = self.collection.count()
        
        if total_count == 0:
            return {"documents": [], "metadatas": [], "ids": []}
        
        # Determine sample size
        sample_size = min(n, total_count)
        
        # Generate random offsets
        if sample_size == total_count:
            # Just get everything
            return self.collection.get(where=filter, limit=total_count)
        
        # Generate unique random offsets
        random_offsets = sorted(random.sample(range(total_count), sample_size))
        
        # Fetch documents at these offsets
        sampled_docs = []
        sampled_metas = []
        sampled_ids = []
        
        for offset in random_offsets:
            result = self.collection.get(
                where=filter,
                limit=1,
                offset=offset
            )
            
            if result["ids"]:
                sampled_ids.extend(result["ids"])
                sampled_docs.extend(result["documents"]) 
                sampled_metas.extend(result["metadatas"])
        
        return {
            "documents": sampled_docs,
            "metadatas": sampled_metas,
            "ids": sampled_ids
        }
    
    def get_by_year(self, year: int) -> dict:
        """
        Get all documents from a specific year.
        
        Args:
            year: Year to filter by
            
        Returns:
            Dict with keys: documents, metadatas, ids
        """
        return self.collection.get(
            where={"year": year},
            limit=10000
        )
    
    def get_by_entity(self, entity: str, limit: int = 100) -> dict:
        """
        Get all documents related to a specific entity.
        
        Args:
            entity: Entity name to filter by
            limit: Maximum number of results
            
        Returns:
            Dict with keys: documents, metadatas, ids
        """
        return self.collection.get(
            where={"entity": entity},
            limit=limit
        )
    
    def get_stats(self) -> dict:
        """Get collection statistics."""
        count = self.collection.count()
        
        # Sample to get metadata variety
        sample = self.collection.get(limit=1000)
        
        years = set()
        entities = set()
        source_types = set()
        
        for meta in sample["metadatas"]:
            if "year" in meta:
                years.add(meta["year"])
            if "entity" in meta:
                entities.add(meta["entity"])
            if "source_type" in meta:
                source_types.add(meta["source_type"])
        
        return {
            "total_documents": count,
            "years": sorted(list(years)),
            "entities": sorted(list(entities))[:20],  # Top 20
            "source_types": sorted(list(source_types)),
            "collection_name": self.collection_name
        }
    
    def delete_collection(self):
        """Delete the entire collection. Use with caution!"""
        self.client.delete_collection(self.collection_name)
        logger.warning(f"Deleted collection: {self.collection_name}")
    
    def clear(self):
        """Clear all documents from collection but keep collection."""
        ids = self.collection.get(limit=10000)["ids"]
        if ids:
            self.collection.delete(ids=ids)
            logger.info(f"Cleared {len(ids)} documents from collection")