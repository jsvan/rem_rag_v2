"""Tests for READING cycle"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from rem_rag_backend.src.llm.openai_client import LLMClient, StructuredLLMClient
from rem_rag_backend.src.vector_store import REMVectorStore
from rem_rag_backend.src.data_processing.chunker import SmartChunker
from rem_rag_backend.src.core.reading_cycle import ReadingCycle


async def test_reading_cycle():
    """Test the full READING cycle on sample articles"""
    
    # Initialize components
    llm = LLMClient()
    structured_llm = StructuredLLMClient()
    vector_store = REMVectorStore(collection_name="test_reading_cycle")
    vector_store.clear()
    chunker = SmartChunker(chunk_size=500)  # Smaller chunks for testing
    
    reading_cycle = ReadingCycle(llm, structured_llm, vector_store, chunker)
    
    # Create test articles
    test_articles = [
        {
            "text": """The Marshall Plan: America's Bold Gambit
            
            In 1948, Secretary of State George Marshall proposed an unprecedented 
            economic recovery program for war-torn Europe. The Marshall Plan would 
            channel billions of dollars to rebuild European economies, but it had a 
            dual purpose: economic recovery and containing Soviet influence.
            
            Stalin saw the plan as American imperialism and forbade Eastern European 
            nations from participating. This decision would deepen the division of 
            Europe. Western Europe embraced the aid, leading to rapid economic growth 
            and stronger ties with the United States.
            
            The plan's success went beyond economics. It created a model for American 
            foreign aid and established the principle that economic stability was 
            essential for political stability. By 1952, European industrial production 
            had surpassed pre-war levels.""",
            "title": "The Marshall Plan: America's Bold Gambit",
            "year": 1948,
            "article_id": "marshall_1948"
        },
        {
            "text": """NATO: The Atlantic Alliance
            
            The North Atlantic Treaty Organization emerged in 1949 as a direct response 
            to Soviet expansionism. For the first time in its history, the United States 
            committed to a peacetime military alliance. Article 5 established that an 
            attack on one member would be considered an attack on all.
            
            This revolutionary commitment reflected how profoundly World War II and the 
            emerging Cold War had changed American thinking about security. Isolationism 
            was dead. The Atlantic Ocean no longer provided security in the nuclear age.
            
            Stalin viewed NATO as an aggressive alliance aimed at the Soviet Union. The 
            formation of NATO would accelerate the militarization of the Cold War and 
            lead to the creation of the Warsaw Pact in 1955.""",
            "title": "NATO: The Atlantic Alliance",
            "year": 1949,
            "article_id": "nato_1949"
        }
    ]
    
    # Process articles chronologically
    print("Processing articles through READING cycle...")
    stats = await reading_cycle.process_articles_chronologically(
        test_articles,
        max_concurrent=1  # Process sequentially for testing
    )
    
    print(f"\nProcessing statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Check what's in the vector store
    store_stats = vector_store.get_stats()
    print(f"\nVector store statistics:")
    print(f"  Total documents: {store_stats['total_documents']}")
    print(f"  Years: {store_stats['years']}")
    print(f"  Sample entities: {store_stats['entities'][:10]}")
    
    # Track entity evolution
    print("\nEntity evolution examples:")
    
    # Check Stalin's evolution
    stalin_evolution = reading_cycle.get_entity_evolution("Stalin", limit=10)
    print(f"\nStalin knowledge evolution ({len(stalin_evolution)} nodes):")
    for node in stalin_evolution:
        print(f"  {node['year']} ({node['node_type']}): {node['text'][:100]}...")
    
    # Check Marshall Plan evolution
    marshall_evolution = reading_cycle.get_entity_evolution("Marshall Plan", limit=10)
    print(f"\nMarshall Plan evolution ({len(marshall_evolution)} nodes):")
    for node in marshall_evolution:
        print(f"  {node['year']} ({node['node_type']}): {node['text'][:100]}...")
    
    # Query for synthesis examples
    syntheses = vector_store.collection.get(
        where={"node_type": "synthesis"},
        limit=5
    )
    
    print(f"\nExample syntheses generated:")
    for i, (doc, meta) in enumerate(zip(syntheses["documents"], syntheses["metadatas"])):
        print(f"\n{i+1}. {meta.get('synthesis_type', 'entity')} synthesis:")
        print(f"   Year: {meta.get('year')}")
        print(f"   Text: {doc[:200]}...")
    
    # Clean up
    vector_store.delete_collection()
    
    print("\nTest completed!")


async def test_chronological_ordering():
    """Test that chronological processing maintains temporal coherence"""
    
    # Initialize components
    llm = LLMClient()
    structured_llm = StructuredLLMClient()
    vector_store = REMVectorStore(collection_name="test_chronological")
    vector_store.clear()
    chunker = SmartChunker(chunk_size=300)
    
    reading_cycle = ReadingCycle(llm, structured_llm, vector_store, chunker)
    
    # Create articles that reference each other
    articles = [
        {
            "text": "In 1950, North Korea invaded South Korea, starting the Korean War.",
            "title": "Korean War Begins",
            "year": 1950,
            "article_id": "korea_1950"
        },
        {
            "text": "The Korean War ended in 1953 with an armistice, dividing Korea permanently.",
            "title": "Korean Armistice",
            "year": 1953,
            "article_id": "korea_1953"
        },
        {
            "text": "Looking back at the Korean War, it established the pattern for limited wars.",
            "title": "Korean War Retrospective",
            "year": 1960,
            "article_id": "korea_retro_1960"
        }
    ]
    
    # Process chronologically
    await reading_cycle.process_articles_chronologically(articles)
    
    # The 1960 article should have syntheses referring to earlier knowledge
    results = vector_store.collection.get(
        where={
            "year": 1960,
            "node_type": "synthesis"
        }
    )
    
    print(f"Found {len(results['ids'])} syntheses from 1960")
    assert len(results['ids']) > 0, "Should have syntheses in 1960 referring to earlier knowledge"
    
    # Clean up
    vector_store.delete_collection()
    
    print("Chronological ordering test passed!")


async def main():
    """Run all tests"""
    print("Testing READING cycle...")
    await test_reading_cycle()
    
    print("\n" + "="*70 + "\n")
    
    print("Testing chronological ordering...")
    await test_chronological_ordering()


if __name__ == "__main__":
    asyncio.run(main())