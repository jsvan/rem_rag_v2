"""Tests for entity processor"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.llm.openai_client import StructuredLLMClient
from src.vector_store import REMVectorStore
from src.core.entity_processor import EntityProcessor


async def test_entity_extraction():
    """Test basic entity extraction and processing"""
    
    # Initialize components
    llm_client = StructuredLLMClient()
    vector_store = REMVectorStore(collection_name="test_entities")
    vector_store.clear()  # Start fresh
    
    processor = EntityProcessor(llm_client, vector_store)
    
    # Test text about Cuban Missile Crisis
    test_chunk = """
    In October 1962, President Kennedy faced his greatest challenge when 
    Khrushchev placed nuclear missiles in Cuba. The Soviet Union's bold move 
    brought the world to the brink of nuclear war. Castro's Cuba became the 
    flashpoint of the Cold War.
    """
    
    print("Processing first chunk...")
    stats1 = await processor.process_chunk_entities(
        chunk_text=test_chunk,
        year=1962,
        article_id="test_001"
    )
    
    print(f"First chunk stats: {stats1}")
    
    # Process a similar chunk to test redundancy detection
    similar_chunk = """
    The Cuban Missile Crisis of 1962 saw Kennedy and Khrushchev in a 
    dangerous standoff. Soviet missiles in Cuba threatened the United States 
    directly. The crisis brought nuclear war closer than ever before.
    """
    
    print("\nProcessing similar chunk...")
    stats2 = await processor.process_chunk_entities(
        chunk_text=similar_chunk,
        year=1962,
        article_id="test_002"
    )
    
    print(f"Second chunk stats: {stats2}")
    
    # Check what's in the vector store
    all_entities = vector_store.get_stats()
    print(f"\nVector store stats: {all_entities}")
    
    # Query specific entity
    kennedy_knowledge = vector_store.query(
        "leadership crisis",
        filter={"entity": "Kennedy"},
        k=10
    )
    
    print(f"\nFound {len(kennedy_knowledge['documents'])} items about Kennedy")
    
    # Check node types
    syntheses = vector_store.collection.get(
        where={"node_type": "synthesis"},
        limit=100
    )
    
    redundant = vector_store.collection.get(
        where={"node_type": "learning_nothing"},
        limit=100
    )
    
    print(f"\nNode type counts:")
    print(f"- Valuable syntheses: {len(syntheses['ids'])}")
    print(f"- Redundant learnings: {len(redundant['ids'])}")
    
    # Clean up
    vector_store.delete_collection()
    
    print("\nTest completed!")


async def test_batch_processing():
    """Test batch processing of multiple chunks"""
    
    # Initialize components
    llm_client = StructuredLLMClient()
    vector_store = REMVectorStore(collection_name="test_batch")
    vector_store.clear()
    
    processor = EntityProcessor(llm_client, vector_store)
    
    # Create test chunks
    chunks = [
        {
            "text": "NATO's formation in 1949 created a Western alliance against Soviet expansion.",
            "year": 1949,
            "article_id": "batch_001"
        },
        {
            "text": "The Marshall Plan rebuilt Europe economically while containing communism.",
            "year": 1948,
            "article_id": "batch_002"
        },
        {
            "text": "Stalin's response to NATO was the Warsaw Pact, dividing Europe.",
            "year": 1955,
            "article_id": "batch_003"
        }
    ]
    
    print("Batch processing chunks...")
    total_stats = await processor.batch_process_chunks(chunks, max_concurrent=2)
    
    print(f"\nBatch processing stats: {total_stats}")
    
    # Clean up
    vector_store.delete_collection()


async def main():
    """Run all tests"""
    print("Testing entity extraction and implant synthesis...")
    await test_entity_extraction()
    
    print("\n" + "="*50 + "\n")
    
    print("Testing batch processing...")
    await test_batch_processing()


if __name__ == "__main__":
    asyncio.run(main())