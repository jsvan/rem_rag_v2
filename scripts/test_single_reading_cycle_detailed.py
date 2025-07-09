#!/usr/bin/env python3
"""Test script that shows EVERY step of processing a single article"""

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

# Set up very detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_single_article_detailed():
    """Test the complete READING cycle on ONE article with full details."""
    
    print("\n" + "="*80)
    print("TESTING SINGLE ARTICLE READING CYCLE - DETAILED VIEW")
    print("="*80 + "\n")
    
    # Initialize components
    llm = LLMClient()
    structured_llm = StructuredLLMClient()
    vector_store = REMVectorStore(collection_name="test_single_article_detailed")
    chunker = SentenceAwareChunker(max_words=300, min_chars=150)
    
    # Clear any existing data
    print("🧹 Clearing vector store...")
    vector_store.clear()
    
    # Create reading cycle
    reading_cycle = ReadingCycle(
        llm_client=llm,
        structured_llm=structured_llm,
        vector_store=vector_store,
        chunker=chunker
    )
    
    # Load just ONE article from 1922
    print("\n📚 Loading 1922 articles...")
    articles_1922 = load_years_data([1922])
    
    # Take the first article
    article = articles_1922[0]
    print(f"\n📄 Selected article: {article['title']}")
    print(f"   Article ID: {article['article_id']}")
    print(f"   Year: {article['year']}")
    print(f"   Text length: {len(article['text'])} characters")
    print(f"   Word count: {len(article['text'].split())} words")
    
    # Show chunking process
    print("\n" + "-"*60)
    print("STEP 1: CHUNKING")
    print("-"*60)
    
    chunks = chunker.chunk_article(article)
    print(f"\n✂️ Created {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"\n📦 Chunk {i+1}:")
        print(f"   Words: {chunk['metadata']['word_count']}")
        print(f"   Characters: {chunk['metadata']['char_count']}")
        print(f"   Preview: {chunk['text'][:150]}...")
    
    if len(chunks) > 3:
        print(f"\n   ... and {len(chunks) - 3} more chunks")
    
    # Process the article with detailed logging
    print("\n" + "-"*60)
    print("STEP 2: PROCESSING ARTICLE")
    print("-"*60)
    
    # Manually process first chunk to show details
    first_chunk = chunks[0]
    print(f"\n🔍 Processing first chunk in detail...")
    print(f"   Chunk text: {first_chunk['text'][:200]}...")
    
    # Extract entities from first chunk
    print("\n📋 Extracting entities...")
    entity_result = await structured_llm.extract_entities(
        first_chunk['text'], 
        "List the key entities and abstract concepts in this passage. For each, write a complete paragraph that includes the entity/concept name and explains what we learn about it from this passage."
    )
    
    print(f"\n✨ Found {len(entity_result)} entities/concepts:")
    for i, entity_data in enumerate(entity_result):
        print(f"\n   Entity {i+1}: {entity_data['entity']}")
        print(f"   Learning: {entity_data['learning'][:200]}...")
    
    # Now process the full article
    print("\n" + "-"*60)
    print("STEP 3: FULL ARTICLE PROCESSING")
    print("-"*60)
    
    stats = await reading_cycle.process_article(article)
    
    print(f"\n📊 Processing Results:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Total entities extracted: {stats['total_entities']}")
    print(f"   Entity syntheses stored: {stats['valuable_syntheses']}")
    print(f"   Chunk syntheses stored: {stats['chunk_syntheses']}")
    print(f"   Processing time: {stats['processing_time']:.1f} seconds")
    
    # Show what's in the database
    print("\n" + "-"*60)
    print("STEP 4: DATABASE CONTENTS")
    print("-"*60)
    
    # Get all nodes
    all_results = vector_store.search("", k=1000)  # Empty query gets all
    
    # Group by node type
    nodes_by_type = {}
    for doc, meta, doc_id in zip(all_results['documents'], 
                                 all_results['metadatas'], 
                                 all_results['ids']):
        node_type = meta.get('node_type', 'unknown')
        if node_type not in nodes_by_type:
            nodes_by_type[node_type] = []
        nodes_by_type[node_type].append((doc, meta, doc_id))
    
    print(f"\n📚 Total nodes stored: {len(all_results['ids'])}")
    for node_type, nodes in sorted(nodes_by_type.items()):
        print(f"   {node_type}: {len(nodes)} nodes")
    
    # Show examples of each node type
    for node_type, nodes in sorted(nodes_by_type.items()):
        print(f"\n🏷️  {node_type.upper()} EXAMPLES:")
        for i, (doc, meta, doc_id) in enumerate(nodes[:2]):  # Show first 2 of each type
            print(f"\n   Example {i+1} (ID: {doc_id[:8]}...):")
            
            # Show relevant metadata
            if node_type == 'chunk':
                print(f"   Article: {meta.get('article_title', 'Unknown')}")
                print(f"   Chunk index: {meta.get('chunk_index', 'Unknown')}")
                print(f"   Word count: {meta.get('word_count', 'Unknown')}")
            elif node_type == 'learning':
                print(f"   Entity: {meta.get('entity', 'Unknown')}")
                print(f"   Source: {meta.get('source_type', 'Unknown')}")
            elif node_type == 'synthesis':
                print(f"   Type: {meta.get('synthesis_type', 'Unknown')}")
                print(f"   Entity: {meta.get('entity', 'N/A')}")
                
            print(f"   Content: {doc[:200]}...")
    
    # Test search functionality
    print("\n" + "-"*60)
    print("STEP 5: TEST SEARCH")
    print("-"*60)
    
    # Search for a key concept from the article
    search_query = "democracy foreign policy"
    print(f"\n🔍 Searching for: '{search_query}'")
    
    search_results = vector_store.search(search_query, k=5)
    print(f"   Found {len(search_results['documents'])} relevant results")
    
    for i, (doc, meta) in enumerate(zip(search_results['documents'][:3], 
                                       search_results['metadatas'][:3])):
        print(f"\n   Result {i+1}:")
        print(f"   Type: {meta.get('node_type', 'Unknown')}")
        print(f"   Preview: {doc[:150]}...")
    
    # Clean up
    print("\n" + "-"*60)
    print("CLEANUP")
    print("-"*60)
    print("🧹 Clearing test collection...")
    vector_store.clear()
    print("✅ Test complete!")


if __name__ == "__main__":
    asyncio.run(test_single_article_detailed())