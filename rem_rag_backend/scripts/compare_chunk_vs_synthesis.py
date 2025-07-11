#!/usr/bin/env python3
"""
Compare original chunks with their generated synthesis to see if synthesis adds value.
"""

import asyncio
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from rem_rag_backend.src.core.implant import implant_knowledge
from rem_rag_backend.src.vector_store.chromadb_store import REMVectorStore
from rem_rag_backend.src.llm.openai_client import LLMClient
from rem_rag_backend.src.config import LLM_MODEL, NEIGHBORS_COUNT


async def compare_chunk_synthesis():
    """Compare chunks with their synthesis to evaluate value added."""
    print("🔍 Comparing Original Chunks vs Generated Synthesis")
    print("=" * 80)
    
    # Initialize components
    vector_store = REMVectorStore()
    llm_client = LLMClient(model=LLM_MODEL)
    
    # Test chunk 1 - Specific event
    chunk1 = """The humanitarian intervention in Kosovo revealed a fundamental paradox: 
NATO's bombing campaign, intended to stop ethnic cleansing, initially accelerated 
the very atrocities it sought to prevent. Serbian forces intensified their campaign 
against Kosovo Albanians after the bombing began, creating a massive refugee crisis 
that destabilized neighboring countries."""
    
    # Test chunk 2 - General principle
    chunk2 = """This pattern reflects a broader dilemma in humanitarian intervention. 
Military action taken to protect civilians can paradoxically endanger them by 
escalating conflicts. Rebel groups may deliberately provoke atrocities to attract 
international support, creating moral hazard."""
    
    # Test chunk 3 - Policy debate
    chunk3 = """The debate continues to divide policymakers. Some argue the moral 
imperative to prevent genocide overrides sovereignty concerns. Others warn that 
interventions often fail because they address symptoms rather than underlying 
political conflicts."""
    
    test_chunks = [
        ("Kosovo Specific Event", chunk1),
        ("General Intervention Pattern", chunk2),
        ("Policy Debate", chunk3)
    ]
    
    for i, (chunk_name, chunk_text) in enumerate(test_chunks):
        print(f"\n{'='*20} TEST {i+1}: {chunk_name} {'='*20}")
        
        # Show original chunk
        print("\n📄 ORIGINAL CHUNK:")
        print("-" * 60)
        print(chunk_text)
        print(f"\nWord count: {len(chunk_text.split())}")
        
        # Get what existing knowledge this relates to
        print("\n🔗 RELATED EXISTING KNOWLEDGE:")
        print("-" * 60)
        
        existing_results = vector_store.query(
            text=chunk_text,
            k=3,
            filter={"year": {"$lt": 2000}}  # Earlier knowledge
        )
        
        if existing_results['documents']:
            for j, (text, metadata) in enumerate(zip(existing_results['documents'][:2], 
                                                    existing_results['metadatas'][:2])):
                print(f"\n{j+1}. {metadata.get('node_type')} ({metadata.get('year')})")
                print(f"   {text[:150]}...")
        else:
            print("No related knowledge found")
        
        # Generate synthesis through implant
        chunk_metadata = {
            "year": 2000,
            "article_id": f"test_{i}",
            "title": chunk_name,
            "node_type": "chunk",
            "generation_depth": 0
        }
        
        implant_result = await implant_knowledge(
            new_content=chunk_text,
            vector_store=vector_store,
            llm_client=llm_client,
            metadata=chunk_metadata,
            context_filter={"year": {"$lt": 2000}},
            k=NEIGHBORS_COUNT
        )
        
        # Show synthesis if generated
        if implant_result['is_valuable'] and implant_result['synthesis']:
            print("\n✨ GENERATED SYNTHESIS:")
            print("-" * 60)
            print(implant_result['synthesis'])
            print(f"\nWord count: {len(implant_result['synthesis'].split())}")
            
            # Analyze what's new/different
            print("\n📊 VALUE ANALYSIS:")
            print("-" * 60)
            
            # Check for new connections
            chunk_lower = chunk_text.lower()
            synthesis_lower = implant_result['synthesis'].lower()
            
            # Key terms from synthesis not in original
            synthesis_words = set(synthesis_lower.split())
            chunk_words = set(chunk_lower.split())
            new_terms = synthesis_words - chunk_words
            
            # Filter out common words
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                          'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                          'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                          'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
                          'those', 'such', 'where', 'when', 'while', 'which', 'who', 'whom'}
            
            significant_new = [word for word in new_terms if word not in common_words and len(word) > 3]
            
            print(f"New significant terms introduced: {', '.join(significant_new[:10])}")
            
            # Check if synthesis connects to broader patterns
            pattern_words = ['pattern', 'illustrates', 'demonstrates', 'reflects', 'underscores',
                           'exemplifies', 'reveals', 'suggests', 'indicates', 'implies']
            uses_pattern_language = any(word in synthesis_lower for word in pattern_words)
            
            print(f"Identifies patterns: {'Yes' if uses_pattern_language else 'No'}")
            
            # Check if synthesis makes historical connections
            historical_refs = ['history', 'historical', 'past', 'previous', 'earlier', 'precedent']
            makes_historical_connections = any(ref in synthesis_lower for ref in historical_refs)
            
            print(f"Makes historical connections: {'Yes' if makes_historical_connections else 'No'}")
            
            # Check compression ratio
            compression_ratio = len(chunk_text) / len(implant_result['synthesis'])
            print(f"Compression ratio: {compression_ratio:.2f}x")
            
            # Does it generalize from specific to pattern?
            if 'kosovo' in chunk_lower and 'kosovo' not in synthesis_lower:
                print("Generalizes: Yes (moves from Kosovo-specific to general pattern)")
            elif any(word in synthesis_lower for word in ['broader', 'general', 'overall', 'pattern']):
                print("Generalizes: Yes (explicitly states broader pattern)")
            else:
                print("Generalizes: No (stays at same level of specificity)")
                
        else:
            print("\n❌ No synthesis generated (content may be too novel or isolated)")
    
    # Now query to see how chunks vs synthesis retrieve
    print("\n\n" + "="*80)
    print("🎯 RETRIEVAL COMPARISON")
    print("="*80)
    
    test_query = "What are the unintended consequences of military intervention?"
    
    print(f"\nQuery: '{test_query}'")
    
    # Get results
    results = vector_store.query(test_query, k=20)
    
    # Analyze positions of chunks vs synthesis
    chunk_positions = []
    synthesis_positions = []
    
    for i, metadata in enumerate(results['metadatas']):
        if metadata.get('node_type') == 'chunk' and 'test_' in metadata.get('article_id', ''):
            chunk_positions.append(i + 1)
        elif metadata.get('node_type') == 'synthesis' and 'test_' in metadata.get('article_id', ''):
            synthesis_positions.append(i + 1)
    
    print(f"\nOriginal chunk positions: {chunk_positions}")
    print(f"Synthesis positions: {synthesis_positions}")
    
    if synthesis_positions and chunk_positions:
        if min(synthesis_positions) < min(chunk_positions):
            print("\n✅ Synthesis retrieves better than original chunks!")
        else:
            print("\n❌ Original chunks retrieve better than synthesis")


if __name__ == "__main__":
    asyncio.run(compare_chunk_synthesis())