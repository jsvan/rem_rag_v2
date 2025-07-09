#!/usr/bin/env python3
"""Clear/reset the ChromaDB database"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store import REMVectorStore

print("=== Clearing REM RAG Database ===")

# Common collection names
collections = [
    "rem_rag_v2",
    "rem_rag_1920s", 
    "test_sentence_reading",
    "test_entity_resolution"
]

for collection_name in collections:
    try:
        store = REMVectorStore(collection_name=collection_name)
        count = store.collection.count()
        if count > 0:
            print(f"\nClearing {collection_name}: {count} documents")
            store.clear()
            print(f"✓ {collection_name} cleared")
        else:
            print(f"✓ {collection_name} is already empty")
    except Exception as e:
        print(f"✗ Could not clear {collection_name}: {e}")

print("\n✓ Database cleanup complete!")