"""Tests for the vector store module"""

import pytest
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.vector_store import REMVectorStore


def test_vector_store_basic():
    """Test basic vector store operations"""
    # Create test instance
    store = REMVectorStore(collection_name="test_rem_rag")
    
    # Clear any existing data
    store.clear()
    
    # Test data
    texts = [
        "The Cuban Missile Crisis was a pivotal moment in the Cold War.",
        "Khrushchev's decision to place missiles in Cuba changed everything.",
        "Nuclear deterrence theory evolved significantly after 1962."
    ]
    
    metadata = [
        {
            "year": 1962,
            "entity": "Cuban Missile Crisis",
            "source_type": "article",
            "generation_depth": 0
        },
        {
            "year": 1962,
            "entity": "Khrushchev",
            "source_type": "article",
            "generation_depth": 0
        },
        {
            "year": 1963,
            "entity": "Nuclear Deterrence",
            "source_type": "synthesis",
            "generation_depth": 1
        }
    ]
    
    # Test add
    ids = store.add(texts, metadata)
    assert len(ids) == 3
    
    # Test query
    results = store.query("nuclear weapons", k=2)
    assert len(results["documents"]) <= 2
    assert all(key in results for key in ["documents", "metadatas", "distances", "ids"])
    
    # Test filter by year
    results = store.query("missiles", k=5, filter={"year": 1962})
    assert all(m["year"] == 1962 for m in results["metadatas"])
    
    # Test get_by_year
    results = store.get_by_year(1962)
    assert len(results["documents"]) == 2
    
    # Test sample
    results = store.sample(2)
    assert len(results["documents"]) == 2
    
    # Test stats
    stats = store.get_stats()
    assert stats["total_documents"] == 3
    assert 1962 in stats["years"]
    assert "Khrushchev" in stats["entities"]
    
    # Cleanup
    store.delete_collection()
    
    print("All tests passed!")


if __name__ == "__main__":
    test_vector_store_basic()