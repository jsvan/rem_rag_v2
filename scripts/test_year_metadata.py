#!/usr/bin/env python3
"""
Test that year metadata is properly added to all node types.
"""

import os
import sys
import json
from datetime import datetime

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.vector_store import REMVectorStore
from src.llm import LLMClient
from src.core.implant import implant_knowledge_sync
from src.core.rem_cycle import REMCycle


def test_year_metadata():
    """Test year metadata across different node types"""
    print("🧪 Testing Year Metadata Implementation")
    print("=" * 50)
    
    # Initialize components
    llm = LLMClient()
    store = REMVectorStore()
    rem = REMCycle(llm, store)
    
    # Test data
    test_year = 2005
    test_content = "The Iraq War continues to dominate American foreign policy discussions."
    
    # 1. Test chunk node
    print("\n1️⃣ Testing CHUNK node...")
    chunk_metadata = {
        'node_type': 'chunk',
        'year': test_year,
        'article_title': 'Test Article',
        'test_type': 'year_metadata_test'
    }
    
    result = implant_knowledge_sync(
        new_content=test_content,
        vector_store=store,
        llm_client=llm,
        metadata=chunk_metadata,
        k=3
    )
    
    # Verify chunk was stored with year
    chunk_check = store.collection.get(ids=[result['original_id']])
    if chunk_check['metadatas'] and chunk_check['metadatas'][0].get('year') == test_year:
        print(f"  ✅ Chunk node has correct year: {test_year}")
    else:
        print(f"  ❌ Chunk node missing year or incorrect: {chunk_check['metadatas']}")
    
    # Check if synthesis was created
    if result['is_valuable'] and result['synthesis_id']:
        synth_check = store.collection.get(ids=[result['synthesis_id']])
        if synth_check['metadatas'] and synth_check['metadatas'][0].get('year') == test_year:
            print(f"  ✅ Synthesis node inherits year: {test_year}")
        else:
            print(f"  ❌ Synthesis node missing year: {synth_check['metadatas']}")
    
    # 2. Test summary node
    print("\n2️⃣ Testing SUMMARY node...")
    summary_metadata = {
        'node_type': 'summary',
        'year': test_year,
        'article_title': 'Test Summary Article',
        'test_type': 'year_metadata_test'
    }
    
    summary_content = "American foreign policy in the Middle East faces unprecedented challenges."
    
    result = implant_knowledge_sync(
        new_content=summary_content,
        vector_store=store,
        llm_client=llm,
        metadata=summary_metadata,
        k=3
    )
    
    summary_check = store.collection.get(ids=[result['original_id']])
    if summary_check['metadatas'] and summary_check['metadatas'][0].get('year') == test_year:
        print(f"  ✅ Summary node has correct year: {test_year}")
    else:
        print(f"  ❌ Summary node missing year: {summary_check['metadatas']}")
    
    # 3. Test learning node
    print("\n3️⃣ Testing LEARNING node...")
    learning_metadata = {
        'node_type': 'learning',
        'year': test_year,
        'entity': 'NATO',
        'article_title': 'Test Learning Article',
        'test_type': 'year_metadata_test'
    }
    
    learning_content = "About NATO: The alliance struggles to define its post-Cold War mission."
    
    result = implant_knowledge_sync(
        new_content=learning_content,
        vector_store=store,
        llm_client=llm,
        metadata=learning_metadata,
        k=3
    )
    
    learning_check = store.collection.get(ids=[result['original_id']])
    if learning_check['metadatas'] and learning_check['metadatas'][0].get('year') == test_year:
        print(f"  ✅ Learning node has correct year: {test_year}")
    else:
        print(f"  ❌ Learning node missing year: {learning_check['metadatas']}")
    
    # 4. Test REM node
    print("\n4️⃣ Testing REM node...")
    print("  Running mini REM cycle...")
    
    # First add some test nodes for REM to sample
    for i in range(3):
        store.add(
            [f"Test content {i} for REM sampling"],
            [{
                'node_type': 'chunk',
                'year': test_year - i,
                'test_type': 'rem_sample',
                'timestamp': datetime.now().isoformat()
            }]
        )
    
    # Run REM cycle
    rem_ids = rem.run_cycle(current_year=test_year)
    
    if rem_ids:
        rem_check = store.collection.get(ids=[rem_ids[0]])
        if rem_check['metadatas'] and rem_check['metadatas'][0].get('year') == test_year:
            print(f"  ✅ REM node has correct year: {test_year}")
            print(f"     Year range: {rem_check['metadatas'][0].get('year_min')} - {rem_check['metadatas'][0].get('year_max')}")
        else:
            print(f"  ❌ REM node missing year: {rem_check['metadatas']}")
    else:
        print("  ⚠️  No REM nodes created")
    
    # 5. Clean up test data
    print("\n🧹 Cleaning up test data...")
    test_filter = {"test_type": {"$in": ["year_metadata_test", "rem_sample"]}}
    test_results = store.collection.get(where=test_filter, limit=100)
    if test_results['ids']:
        store.collection.delete(ids=test_results['ids'])
        print(f"  Deleted {len(test_results['ids'])} test nodes")
    
    print("\n✅ Year metadata test complete!")


if __name__ == "__main__":
    test_year_metadata()