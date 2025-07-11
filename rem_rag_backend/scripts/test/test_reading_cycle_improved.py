#!/usr/bin/env python3
"""
Test the reading cycle with the improved synthesis prompt to see the quality of generated synthesis nodes.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from rem_rag_backend.src.core.reading_cycle import ReadingCycle
from rem_rag_backend.src.vector_store.chromadb_store import REMVectorStore
from rem_rag_backend.src.llm.openai_client import LLMClient, StructuredLLMClient
from rem_rag_backend.src.data_processing.sentence_chunker import SentenceAwareChunker
from rem_rag_backend.src.config import LLM_MODEL


async def test_reading_cycle():
    """Test the reading cycle to see synthesis generation quality."""
    print("📚 Testing Reading Cycle with Improved Synthesis Prompt")
    print("=" * 80)
    
    # Initialize components
    vector_store = REMVectorStore()
    llm_client = LLMClient(model=LLM_MODEL)
    structured_llm = StructuredLLMClient(model=LLM_MODEL)  # Use same model
    chunker = SentenceAwareChunker()
    
    reading_cycle = ReadingCycle(
        llm_client=llm_client,
        structured_llm=structured_llm,
        vector_store=vector_store,
        chunker=chunker
    )
    
    # Create a test article
    test_article = {
        "article_id": "test_2000_001",
        "title": "The Paradox of Humanitarian Intervention",
        "year": 2000,
        "text": """The humanitarian intervention in Kosovo revealed fundamental paradoxes in international relations. NATO's bombing campaign, intended to stop ethnic cleansing, initially accelerated the very atrocities it sought to prevent. Serbian forces intensified their campaign against Kosovo Albanians after the bombing began, creating a massive refugee crisis.

This pattern reflects a broader dilemma in humanitarian intervention. Military action taken to protect civilians can paradoxically endanger them by escalating conflicts. Rebel groups may deliberately provoke atrocities to attract international support, creating moral hazard.

The debate continues to divide policymakers. Some argue the moral imperative to prevent genocide overrides sovereignty concerns. Others warn that interventions often fail because they address symptoms rather than underlying political conflicts. Without understanding local dynamics and having clear political objectives, military interventions risk becoming indefinite peacekeeping operations that freeze rather than resolve conflicts.

Historical examples from Rwanda to Bosnia demonstrate this pattern. The international community's delayed response in Rwanda allowed genocide to proceed, while in Bosnia, intervention came too late to prevent ethnic cleansing. These failures highlight how the promise of eventual intervention can discourage compromise and encourage escalation by weaker parties expecting foreign support."""
    }
    
    print(f"\n📄 Test Article: {test_article['title']}")
    print(f"Year: {test_article['year']}")
    print(f"Text length: {len(test_article['text'])} characters")
    
    # Process the article
    print("\n\n🔄 Processing article through reading cycle...")
    print("=" * 80)
    
    stats = await reading_cycle.process_article(test_article)
    
    print(f"\n✅ Processing complete!")
    print(f"Chunks processed: {stats['total_chunks']}")
    print(f"Entities extracted: {stats['total_entities']}")
    print(f"Valuable syntheses: {stats['valuable_syntheses']}")
    print(f"Processing time: {stats['processing_time']:.2f}s")
    
    # Query for synthesis nodes created
    print("\n\n🔍 Examining Generated Synthesis Nodes")
    print("=" * 80)
    
    # Get synthesis nodes from this article
    synthesis_filter = {
        "$and": [
            {"node_type": "synthesis"},
            {"article_id": "test_2000_001"}
        ]
    }
    
    # Query for all synthesis nodes from this article
    all_syntheses = []
    offset = 0
    while True:
        results = vector_store.collection.get(
            where=synthesis_filter,
            limit=100,
            offset=offset
        )
        if not results['documents']:
            break
        all_syntheses.extend(zip(results['documents'], results['metadatas']))
        offset += 100
        if len(results['documents']) < 100:
            break
    
    print(f"\nTotal synthesis nodes generated: {len(all_syntheses)}")
    
    # Display synthesis examples
    if all_syntheses:
        print("\n📝 Synthesis Examples:")
        print("-" * 80)
        
        for i, (text, metadata) in enumerate(all_syntheses[:5]):  # Show first 5
            print(f"\n--- Synthesis {i+1} ---")
            print(f"Type: {metadata.get('synthesis_type', 'unknown')}")
            print(f"Entity: {metadata.get('entity', 'N/A')}")
            print(f"Generation depth: {metadata.get('generation_depth', 0)}")
            print(f"\nText: {text}")
            
            # Check for meta-language
            meta_words = ["reinforces", "extends", "highlights", "aligns with", "builds upon"]
            contains_meta = any(word in text.lower() for word in meta_words)
            print(f"\nContains meta-language: {'YES ⚠️' if contains_meta else 'NO ✅'}")
    
    # Test retrieval quality
    print("\n\n🎯 Testing Retrieval Quality")
    print("=" * 80)
    
    test_query = "What are the paradoxes of humanitarian intervention?"
    results = vector_store.query(test_query, k=10)
    
    # Count node types in results
    node_types = {}
    synthesis_positions = []
    for i, metadata in enumerate(results['metadatas']):
        node_type = metadata.get('node_type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
        if node_type == 'synthesis':
            synthesis_positions.append(i + 1)
    
    print(f"Query: '{test_query}'")
    print(f"\nNode type distribution in top 10:")
    for node_type, count in sorted(node_types.items()):
        print(f"  {node_type}: {count}")
    
    if synthesis_positions:
        print(f"\nSynthesis positions: {synthesis_positions}")
        print("\nSynthesis nodes in results:")
        for i, (text, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
            if metadata.get('node_type') == 'synthesis':
                print(f"\n  Position {i+1}: {text[:150]}...")


if __name__ == "__main__":
    asyncio.run(test_reading_cycle())