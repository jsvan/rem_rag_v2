#!/usr/bin/env python3
"""Step-by-step walkthrough of processing a single article"""

import asyncio
import logging
import json
import os
import sys
from typing import List, Dict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMClient, StructuredLLMClient
from src.vector_store import REMVectorStore
from src.data_processing.sentence_chunker import SentenceAwareChunker
from src.core.entity_processor import EntityProcessor
from src.core.implant import implant_knowledge
from src.utils.data_loader import load_years_data

# Silence HTTP logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("src.vector_store").setLevel(logging.WARNING)

# Set up minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def display_chunk(chunk: Dict, index: int):
    """Display a chunk with formatting."""
    print(f"\n{'='*80}")
    print(f"CHUNK {index + 1}")
    print(f"{'='*80}")
    print(f"Word count: {chunk['metadata']['word_count']}")
    print(f"\nText:")
    print(chunk['text'])
    print(f"{'='*80}")


def display_entities(entities: List[Dict]):
    """Display extracted entities."""
    print(f"\n{'='*80}")
    print(f"EXTRACTED ENTITIES ({len(entities)} found)")
    print(f"{'='*80}")
    
    for i, entity in enumerate(entities):
        print(f"\nEntity {i+1}: {entity['entity']}")
        print(f"-" * 40)
        print(f"Learning:\n{entity['learning']}")


def display_similar_chunks(similar_docs: Dict, query_text: str):
    """Display similar chunks used for context."""
    print(f"\n{'='*80}")
    print("SIMILAR EXISTING KNOWLEDGE")
    print(f"{'='*80}")
    print(f"\nQuerying for similar knowledge about:")
    print(f"{query_text[:200]}...")
    
    if not similar_docs['documents']:
        print("\n[No similar existing knowledge found - this is new information]")
    else:
        print(f"\nFound {len(similar_docs['documents'])} similar pieces:")
        for i, (doc, meta) in enumerate(zip(similar_docs['documents'], similar_docs['metadatas'])):
            print(f"\n{i+1}. Type: {meta.get('node_type', 'unknown')} | Year: {meta.get('year', 'unknown')}")
            print(f"   {doc[:200]}...")


def display_synthesis(synthesis: str, is_valuable: bool):
    """Display synthesis result."""
    print(f"\n{'='*80}")
    print("SYNTHESIS RESULT")
    print(f"{'='*80}")
    print(f"\nSynthesis:")
    print(synthesis)
    print(f"\nValuable? {'YES - Will be stored' if is_valuable else 'NO - Redundant (NOTHING)'}")


async def process_single_article_step_by_step():
    """Process one article with step-by-step display."""
    
    print("\n" + "="*80)
    print("SINGLE ARTICLE PROCESSING - STEP BY STEP")
    print("="*80)
    print("\nThis will walk through processing ONE article, showing:")
    print("1. How text is chunked")
    print("2. Entity extraction from each chunk")
    print("3. Similar knowledge lookup")
    print("4. Synthesis generation")
    print("\nPress Enter to continue at each step...")
    input()
    
    # Initialize components
    print("\nInitializing components...")
    llm = LLMClient()
    structured_llm = StructuredLLMClient()
    vector_store = REMVectorStore(collection_name="test_step_by_step")
    chunker = SentenceAwareChunker(max_words=300, min_chars=150)
    entity_processor = EntityProcessor(structured_llm, vector_store)
    
    # Clear any existing data
    vector_store.clear()
    
    # Load first article from 1922
    articles = load_years_data([1922])
    article = articles[0]
    
    print(f"\n📄 Article: {article['title']}")
    print(f"Year: {article['year']}")
    print(f"Length: {len(article['text'].split())} words")
    input("\nPress Enter to see chunking...")
    
    # STEP 1: Chunking
    chunks = chunker.chunk_article(article)
    print(f"\n✂️  Split into {len(chunks)} chunks")
    
    # Process first 3 chunks only for demo
    demo_chunks = chunks[:3]
    
    for chunk_idx, chunk in enumerate(demo_chunks):
        display_chunk(chunk, chunk_idx)
        input("\nPress Enter to extract entities from this chunk...")
        
        # STEP 2: Entity Extraction
        chunk_text = chunk['text']
        entities = await structured_llm.extract_entities(
            chunk_text,
            "List the key entities and abstract concepts in this passage. For each, write a complete paragraph that includes the entity/concept name and explains what we learn about it from this passage."
        )
        
        display_entities(entities)
        
        # Process each entity
        for entity_idx, entity_data in enumerate(entities):
            entity_name = entity_data['entity']
            learning = entity_data['learning']
            
            input(f"\nPress Enter to process entity {entity_idx + 1}/{len(entities)}: {entity_name}...")
            
            # STEP 3: Find similar existing knowledge
            similar = vector_store.query(
                text=learning,
                filter={"entity": entity_name},
                k=3
            )
            
            display_similar_chunks(similar, learning)
            
            # STEP 4: Generate synthesis
            input("\nPress Enter to generate synthesis...")
            
            implant_result = await implant_knowledge(
                new_content=learning,
                vector_store=vector_store,
                llm_client=llm,
                metadata={
                    "entity": entity_name,
                    "year": article['year'],
                    "article_id": article['article_id'],
                    "source_type": "entity_synthesis",
                    "node_type": "synthesis",
                    "synthesis_type": "entity_level"
                },
                context_filter={"entity": entity_name},
                k=3
            )
            
            display_synthesis(implant_result['synthesis'], implant_result['is_valuable'])
            
            # Always store the learning itself
            vector_store.add(
                [learning],
                [{
                    "entity": entity_name,
                    "year": article['year'],
                    "article_id": article['article_id'],
                    "node_type": "learning",
                    "source_type": "entity_extraction"
                }]
            )
            print("\n✓ Stored entity learning")
            
            if implant_result['is_valuable']:
                print(f"✓ Stored synthesis with ID: {implant_result['synthesis_id']}")
        
        # STEP 5: Chunk-level synthesis
        input(f"\nPress Enter to generate chunk-level synthesis...")
        
        chunk_similar = vector_store.query(
            text=chunk_text,
            filter={"year": {"$lt": article['year']}},
            k=5
        )
        
        display_similar_chunks(chunk_similar, chunk_text)
        
        chunk_implant = await implant_knowledge(
            new_content=chunk_text,
            vector_store=vector_store,
            llm_client=llm,
            metadata={
                "year": article['year'],
                "article_id": article['article_id'],
                "source_type": "chunk_synthesis",
                "node_type": "synthesis",
                "synthesis_type": "chunk_level"
            },
            context_filter={"year": {"$lt": article['year']}},
            k=5
        )
        
        display_synthesis(chunk_implant['synthesis'], chunk_implant['is_valuable'])
        
        if chunk_implant['is_valuable']:
            print(f"✓ Stored chunk synthesis with ID: {chunk_implant['synthesis_id']}")
        
        if chunk_idx < len(demo_chunks) - 1:
            input(f"\nPress Enter to process next chunk ({chunk_idx + 2}/{len(demo_chunks)})...")
    
    # Summary
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    
    all_nodes = vector_store.collection.get(limit=1000)
    node_types = {}
    for meta in all_nodes['metadatas']:
        node_type = meta.get('node_type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print(f"\nDatabase now contains:")
    for node_type, count in sorted(node_types.items()):
        print(f"  {node_type}: {count}")
    
    print(f"\nTotal nodes: {len(all_nodes['ids'])}")
    
    # Cleanup
    input("\nPress Enter to clean up...")
    vector_store.clear()
    print("✓ Test collection cleared")


if __name__ == "__main__":
    asyncio.run(process_single_article_step_by_step())