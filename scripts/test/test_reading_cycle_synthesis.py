#!/usr/bin/env python3
"""
Test the reading cycle to see how synthesis nodes are generated through the implant process.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.implant import implant_knowledge
from src.vector_store.chromadb_store import REMVectorStore
from src.llm.openai_client import LLMClient
from src.config import LLM_MODEL, NEIGHBORS_COUNT


async def test_reading_synthesis():
    """Test how synthesis nodes are generated during the reading cycle."""
    print("📖 Testing Reading Cycle Synthesis Generation")
    print("=" * 80)
    
    # Initialize components
    vector_store = REMVectorStore()
    llm_client = LLMClient(model=LLM_MODEL)
    
    # Simulate a chunk being processed during reading
    test_chunk = """The humanitarian intervention in Kosovo revealed a fundamental paradox: 
    NATO's bombing campaign, intended to stop ethnic cleansing, initially accelerated 
    the very atrocities it sought to prevent. Serbian forces intensified their campaign 
    against Kosovo Albanians after the bombing began, creating a massive refugee crisis 
    that destabilized neighboring countries."""
    
    chunk_metadata = {
        "year": 2000,
        "article_id": "test_article_001",
        "title": "Lessons from Kosovo",
        "chunk_index": 0,
        "source_type": "chunk",
        "node_type": "chunk",
        "generation_depth": 0,
        "word_count": 50,
        "chunker": "sentence_aware"
    }
    
    print("\n📄 Test Chunk:")
    print("-" * 60)
    print(test_chunk)
    print(f"\nYear: {chunk_metadata['year']}")
    print(f"Article: {chunk_metadata['title']}")
    
    # Step 1: Use implant to store chunk and potentially generate synthesis
    print("\n\n🔗 Implanting chunk (comparing with existing knowledge)...")
    print("=" * 80)
    
    implant_result = await implant_knowledge(
        new_content=test_chunk,
        vector_store=vector_store,
        llm_client=llm_client,
        metadata=chunk_metadata,
        context_filter={"year": {"$lt": 2000}},  # Only look at earlier knowledge
        k=NEIGHBORS_COUNT
    )
    
    print(f"\n✅ Chunk stored: {implant_result['original_id']}")
    print(f"📊 Related nodes found: {implant_result['existing_count']}")
    
    if implant_result['existing_count'] > 0:
        # Query to see what existing knowledge was found
        existing_results = vector_store.query(
            text=test_chunk,
            k=3,
            filter={"year": {"$lt": 2000}}
        )
        
        print("\n📚 Related existing knowledge:")
        print("-" * 60)
        for i, (text, metadata) in enumerate(zip(existing_results['documents'][:3], 
                                                existing_results['metadatas'][:3])):
            print(f"\n{i+1}. Type: {metadata.get('node_type', 'unknown')} | "
                  f"Year: {metadata.get('year', 'unknown')}")
            print(f"   {text[:200]}...")
    
    if implant_result['is_valuable'] and implant_result['synthesis']:
        print("\n\n✨ SYNTHESIS GENERATED!")
        print("=" * 80)
        print("\n🔍 Raw synthesis text:")
        print("-" * 60)
        print(implant_result['synthesis'])
        
        print("\n\n📊 Synthesis node metadata:")
        print("-" * 60)
        # Retrieve the stored synthesis node
        if implant_result['synthesis_id']:
            synthesis_results = vector_store.query(
                text=implant_result['synthesis'],
                k=1
            )
            if synthesis_results['metadatas']:
                syn_meta = synthesis_results['metadatas'][0]
                print(f"Node type: {syn_meta.get('node_type')}")
                print(f"Generation depth: {syn_meta.get('generation_depth')}")
                print(f"Synthesis type: {syn_meta.get('synthesis_type')}")
                print(f"Year: {syn_meta.get('year')}")
    else:
        print("\n\n❌ No synthesis generated (content may be too novel or repetitive)")
    
    # Test 2: Process another chunk that should relate to the first
    print("\n\n" + "="*80)
    print("📄 Testing with a related chunk...")
    print("="*80)
    
    test_chunk2 = """The debate over humanitarian intervention continues to divide policymakers. 
    Some argue that the moral imperative to prevent genocide overrides sovereignty concerns, 
    while others warn that interventions often exacerbate conflicts by providing incentives 
    for rebel groups to provoke international action through deliberate escalation."""
    
    chunk_metadata2 = chunk_metadata.copy()
    chunk_metadata2.update({
        "chunk_index": 1,
        "article_id": "test_article_002",
        "title": "The Intervention Dilemma"
    })
    
    print("\nChunk 2:")
    print(test_chunk2)
    
    implant_result2 = await implant_knowledge(
        new_content=test_chunk2,
        vector_store=vector_store,
        llm_client=llm_client,
        metadata=chunk_metadata2,
        context_filter={"year": {"$lte": 2000}},  # Include current year
        k=NEIGHBORS_COUNT
    )
    
    print(f"\n✅ Chunk stored: {implant_result2['original_id']}")
    print(f"📊 Related nodes found: {implant_result2['existing_count']}")
    
    if implant_result2['is_valuable'] and implant_result2['synthesis']:
        print("\n✨ SYNTHESIS GENERATED!")
        print("-" * 60)
        print(implant_result2['synthesis'])
    
    # Show how synthesis nodes can be queried
    print("\n\n🔍 Querying for synthesis nodes about humanitarian intervention...")
    print("=" * 80)
    
    synthesis_results = vector_store.query(
        text="humanitarian intervention paradox moral hazard",
        k=10,
        filter={"node_type": "synthesis"}
    )
    
    if synthesis_results['documents']:
        print(f"\nFound {len(synthesis_results['documents'])} synthesis nodes")
        for i, (text, metadata) in enumerate(zip(synthesis_results['documents'][:3], 
                                                synthesis_results['metadatas'][:3])):
            print(f"\n{i+1}. Year: {metadata.get('year', 'unknown')}")
            print(f"   {text[:200]}...")


if __name__ == "__main__":
    asyncio.run(test_reading_synthesis())