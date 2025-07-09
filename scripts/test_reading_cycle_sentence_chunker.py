#!/usr/bin/env python3
"""Test script to verify READING cycle with sentence-aware chunking"""

import asyncio
import logging
import json
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMClient, StructuredLLMClient
from src.vector_store import REMVectorStore
from src.data_processing.sentence_chunker import SentenceAwareChunker
from src.core.reading_cycle import ReadingCycle
from src.utils.data_loader import load_years_data
from src.utils.visualization import display_entity_evolution


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_reading_with_sentence_chunker():
    """Test the complete READING cycle with sentence-aware chunking."""
    
    # Initialize components
    llm = LLMClient()
    structured_llm = StructuredLLMClient()
    vector_store = REMVectorStore(collection_name="test_sentence_reading")
    chunker = SentenceAwareChunker(max_words=300, min_chars=150)
    
    # Create reading cycle
    reading_cycle = ReadingCycle(
        llm_client=llm,
        structured_llm=structured_llm,
        vector_store=vector_store,
        chunker=chunker
    )
    
    # Load a few articles from 1922 for testing
    logger.info("Loading 1922 articles...")
    articles_1922 = load_years_data([1922])
    
    # Test with first 3 articles
    test_articles = articles_1922[:3]
    logger.info(f"Testing with {len(test_articles)} articles")
    
    # Display article info
    for article in test_articles:
        logger.info(f"Article: {article['title']} (ID: {article['article_id']})")
    
    # Process articles
    logger.info("\nStarting READING cycle...")
    stats = await reading_cycle.process_articles_chronologically(test_articles)
    
    # Display results
    logger.info("\n=== READING Cycle Results ===")
    logger.info(f"Total articles processed: {stats['total_articles']}")
    logger.info(f"Total chunks created: {stats['total_chunks']}")
    logger.info(f"Total entities extracted: {stats['total_entities']}")
    logger.info(f"Entity syntheses stored: {stats['total_syntheses']}")
    logger.info(f"Chunk syntheses stored: {stats['chunk_syntheses']}")
    logger.info(f"Processing time: {stats['total_time']:.1f} seconds")
    
    # Check what's in the database
    logger.info("\n=== Database Contents ===")
    
    # Get all nodes
    all_results = vector_store.search("", k=500)  # Empty query gets all
    
    # Count by node type
    node_types = {}
    for meta in all_results['metadatas']:
        node_type = meta.get('node_type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    logger.info(f"Total nodes in database: {len(all_results['ids'])}")
    for node_type, count in sorted(node_types.items()):
        logger.info(f"  {node_type}: {count}")
    
    # Sample some chunks to verify sentence-aware chunking
    logger.info("\n=== Sample Chunks (Sentence-Aware) ===")
    chunks = [r for r in zip(all_results['documents'], all_results['metadatas']) 
              if r[1].get('node_type') == 'chunk']
    
    for i, (text, meta) in enumerate(chunks[:3]):
        logger.info(f"\nChunk {i+1}:")
        logger.info(f"  Article: {meta.get('title', 'Unknown')}")
        logger.info(f"  Words: {meta.get('word_count', 'Unknown')}")
        logger.info(f"  Chunker: {meta.get('chunker', 'Unknown')}")
        logger.info(f"  Preview: {text[:200]}...")
    
    # Sample some syntheses
    logger.info("\n=== Sample Syntheses ===")
    syntheses = [r for r in zip(all_results['documents'], all_results['metadatas']) 
                 if r[1].get('node_type') == 'synthesis']
    
    for i, (text, meta) in enumerate(syntheses[:3]):
        logger.info(f"\nSynthesis {i+1}:")
        logger.info(f"  Type: {meta.get('synthesis_type', 'Unknown')}")
        logger.info(f"  Source: {meta.get('source_type', 'Unknown')}")
        logger.info(f"  Text: {text}")
    
    # Test entity evolution tracking
    logger.info("\n=== Entity Evolution Test ===")
    
    # Find some entities that were extracted
    entity_insights = [r for r in zip(all_results['documents'], all_results['metadatas']) 
                      if r[1].get('node_type') == 'entity_insight']
    
    if entity_insights:
        # Get unique entities
        entities = set()
        for _, meta in entity_insights:
            entity_list = meta.get('entities', '[]')
            if isinstance(entity_list, str):
                entity_list = json.loads(entity_list)
            entities.update(entity_list)
        
        logger.info(f"Found {len(entities)} unique entities")
        
        # Track evolution of first entity
        if entities:
            test_entity = list(entities)[0]
            logger.info(f"\nTracking evolution of '{test_entity}':")
            
            evolution = reading_cycle.get_entity_evolution(test_entity, limit=10)
            for node in evolution:
                logger.info(f"  Year {node['year']}, Type: {node['node_type']}")
                logger.info(f"    {node['text'][:150]}...")
    
    # Clean up
    logger.info("\n=== Cleanup ===")
    vector_store.reset()
    logger.info("Test collection cleared")


if __name__ == "__main__":
    asyncio.run(test_reading_with_sentence_chunker())