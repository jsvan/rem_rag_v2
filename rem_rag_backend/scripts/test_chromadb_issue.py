#!/usr/bin/env python3
"""
Test script to diagnose the "Error finding id" issue in ChromaDB.
"""

import os
import sys
import json

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(project_root))

from rem_rag_backend.src.vector_store import REMVectorStore


def test_metadata_issues():
    """Test various metadata scenarios that might cause issues"""
    print("Testing ChromaDB metadata handling...")
    
    store = REMVectorStore(collection_name="test_metadata_issues")
    
    # Test 1: Normal metadata
    print("\n1. Testing normal metadata...")
    try:
        ids = store.add(
            ["Test content 1"],
            [{"node_type": "test", "year": 2000, "title": "Test"}]
        )
        print(f"✅ Success: {ids}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 2: Metadata with None values
    print("\n2. Testing metadata with None values...")
    try:
        ids = store.add(
            ["Test content 2"],
            [{"node_type": "test", "year": 2000, "author": None}]
        )
        print(f"✅ Success: {ids}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 3: Query with filter
    print("\n3. Testing query with year filter...")
    try:
        results = store.query(
            "Test query",
            k=5,
            filter={"year": {"$lt": 2001}}
        )
        print(f"✅ Success: Found {len(results['documents'])} documents")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 4: Large metadata
    print("\n4. Testing large metadata...")
    try:
        large_meta = {
            "node_type": "test",
            "year": 2000,
            "title": "A" * 1000,  # Very long title
            "content_preview": "B" * 5000  # Very long preview
        }
        ids = store.add(
            ["Test content 3"],
            [large_meta]
        )
        print(f"✅ Success: {ids}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 5: Special characters in metadata
    print("\n5. Testing special characters in metadata...")
    try:
        special_meta = {
            "node_type": "test",
            "year": 2000,
            "title": "Test with 'quotes' and \"double quotes\" and \\backslashes"
        }
        ids = store.add(
            ["Test content 4"],
            [special_meta]
        )
        print(f"✅ Success: {ids}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 6: Empty string values
    print("\n6. Testing empty string values...")
    try:
        empty_meta = {
            "node_type": "test",
            "year": 2000,
            "author": "",
            "title": ""
        }
        ids = store.add(
            ["Test content 5"],
            [empty_meta]
        )
        print(f"✅ Success: {ids}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    print("\nTesting complete!")


def test_concurrent_operations():
    """Test concurrent operations that might cause issues"""
    import asyncio
    
    async def add_and_query(store, i):
        """Add a document and immediately query for it"""
        try:
            # Add document
            ids = store.add(
                [f"Concurrent test content {i}"],
                [{"node_type": "test", "year": 2000 + i, "index": i}]
            )
            
            # Immediately query
            results = store.query(
                f"Concurrent test content {i}",
                k=1
            )
            
            return True, None
        except Exception as e:
            return False, str(e)
    
    async def run_concurrent_test():
        store = REMVectorStore(collection_name="test_concurrent")
        
        print("\nTesting concurrent operations...")
        tasks = []
        for i in range(10):
            tasks.append(add_and_query(store, i))
        
        results = await asyncio.gather(*tasks)
        
        successes = sum(1 for success, _ in results if success)
        failures = [(i, err) for i, (success, err) in enumerate(results) if not success]
        
        print(f"Successes: {successes}/10")
        if failures:
            print("Failures:")
            for i, err in failures:
                print(f"  Task {i}: {err}")
    
    asyncio.run(run_concurrent_test())


if __name__ == "__main__":
    print("🔍 Diagnosing ChromaDB 'Error finding id' issue\n")
    
    # Test metadata issues
    test_metadata_issues()
    
    # Test concurrent operations
    test_concurrent_operations()