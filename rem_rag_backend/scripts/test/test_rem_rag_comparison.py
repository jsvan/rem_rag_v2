#!/usr/bin/env python3
"""
Test REM cycle with RAG enhancement - compare answers from 3 nodes vs RAG-enhanced context.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from rem_rag_backend.src.core.rem_cycle import REMCycle
from rem_rag_backend.src.vector_store.chromadb_store import REMVectorStore
from rem_rag_backend.src.llm.openai_client import LLMClient
from rem_rag_backend.src.config import LLM_MODEL


def test_rem_rag_comparison():
    """Test REM cycle comparing original 3-node answer with RAG-enhanced answer."""
    print("🌙 REM Cycle: Comparing Original vs RAG-Enhanced Answers")
    print("=" * 80)
    
    # Initialize components
    vector_store = REMVectorStore()
    llm_client = LLMClient(model=LLM_MODEL)
    rem_cycle = REMCycle(llm_client, vector_store)
    
    # Sample 3 nodes for REM
    print("\n📚 Sampling 3 nodes for REM cycle...")
    print("-" * 60)
    
    samples = rem_cycle._sample_nodes(current_year=None)
    
    if not samples or len(samples) < 3:
        print("❌ Error: Not enough nodes to sample")
        return
    
    # Display the sampled nodes
    print("\n🎲 SAMPLED NODES:")
    print("=" * 80)
    
    for i, sample in enumerate(samples):
        print(f"\n{'='*20} NODE {i+1} {'='*20}")
        print(f"Type: {sample.metadata.get('node_type', 'unknown')}")
        print(f"Year: {sample.metadata.get('year', 'unknown')}")
        if sample.metadata.get('title'):
            print(f"Article: {sample.metadata['title']}")
        if sample.metadata.get('entity'):
            print(f"Entity: {sample.metadata['entity']}")
        print("-" * 60)
        print(sample.content[:500] + "..." if len(sample.content) > 500 else sample.content)
    
    # Generate the implicit question
    print("\n\n❓ GENERATING IMPLICIT QUESTION...")
    print("=" * 80)
    question = rem_cycle._find_implicit_question(samples)
    print(question)
    
    # Generate original synthesis (from just 3 nodes)
    print("\n\n💡 ORIGINAL REM SYNTHESIS (from 3 nodes only):")
    print("=" * 80)
    original_synthesis = rem_cycle._generate_synthesis(samples, question)
    print(f"Question: {question}\n")
    print(f"Synthesis: {original_synthesis}")
    
    # Now do RAG lookup using the question
    print("\n\n🔍 PERFORMING RAG LOOKUP...")
    print("=" * 80)
    
    # Query for related content using the question
    rag_results = vector_store.query(question, k=5)
    
    print(f"Found {len(rag_results['documents'])} additional relevant passages")
    
    # Display RAG results
    print("\n📚 ADDITIONAL CONTEXT FROM RAG:")
    print("-" * 60)
    for i, (text, metadata) in enumerate(zip(rag_results['documents'], rag_results['metadatas'])):
        print(f"\nRAG {i+1}. Type: {metadata.get('node_type')} | Year: {metadata.get('year')}")
        print(f"   {text[:150]}...")
    
    # Create enhanced context: original 3 nodes + 5 RAG results
    print("\n\n🔄 GENERATING RAG-ENHANCED ANSWER...")
    print("=" * 80)
    
    # Prepare enhanced context
    enhanced_context = []
    
    # Add original 3 nodes
    for i, sample in enumerate(samples):
        enhanced_context.append(
            f"Original Node {i+1} ({sample.metadata.get('year', 'Unknown')}, "
            f"{sample.metadata.get('node_type', 'unknown')}):\n{sample.content}"
        )
    
    # Add RAG results
    for i, (text, metadata) in enumerate(zip(rag_results['documents'], rag_results['metadatas'])):
        enhanced_context.append(
            f"Additional Context {i+1} ({metadata.get('year', 'Unknown')}, "
            f"{metadata.get('node_type', 'unknown')}):\n{text}"
        )
    
    # Create enhanced prompt
    enhanced_prompt = f"""You are a wise historian analyzing these passages. 

Question: {question}

Context (3 original passages + 5 additional relevant passages):

{chr(10).join(enhanced_context)}

Based on ALL these passages, provide a comprehensive synthesis that answers the question. 
Draw connections across all the sources to reveal deeper patterns and insights.
Write exactly 1-2 paragraphs."""
    
    rag_synthesis = llm_client.generate_sync(
        prompt=enhanced_prompt,
        temperature=0.7,
        max_tokens=400
    )
    
    print(f"Question: {question}\n")
    print(f"RAG-Enhanced Synthesis: {rag_synthesis}")
    
    # Compare the two syntheses
    print("\n\n📊 COMPARISON ANALYSIS")
    print("=" * 80)
    
    print("\n1️⃣ ORIGINAL SYNTHESIS (3 nodes only):")
    print("-" * 60)
    print(original_synthesis)
    print(f"\nWord count: {len(original_synthesis.split())}")
    
    print("\n\n2️⃣ RAG-ENHANCED SYNTHESIS (3 + 5 nodes):")
    print("-" * 60)
    print(rag_synthesis)
    print(f"\nWord count: {len(rag_synthesis.split())}")
    
    # Analyze differences
    print("\n\n🔬 ANALYSIS:")
    print("-" * 60)
    
    # Check for new concepts
    original_words = set(original_synthesis.lower().split())
    rag_words = set(rag_synthesis.lower().split())
    
    # Remove common words
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                  'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                  'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
                  'those', 'such', 'where', 'when', 'while', 'which', 'who', 'whom',
                  'it', 'its', 'they', 'their', 'them', 'how', 'what', 'why'}
    
    new_concepts = [word for word in (rag_words - original_words) 
                   if word not in common_words and len(word) > 3]
    
    print(f"New concepts in RAG version: {', '.join(new_concepts[:15])}")
    
    # Check depth indicators
    depth_words = ['however', 'moreover', 'furthermore', 'additionally', 'consequently',
                  'therefore', 'thus', 'hence', 'nevertheless', 'nonetheless']
    
    original_depth = sum(1 for word in depth_words if word in original_synthesis.lower())
    rag_depth = sum(1 for word in depth_words if word in rag_synthesis.lower())
    
    print(f"\nDepth indicators - Original: {original_depth}, RAG: {rag_depth}")
    
    # Check for specific examples
    print(f"\nSpecific examples mentioned:")
    examples = ['kosovo', 'rwanda', 'bosnia', 'somalia', 'vietnam', 'iraq', 'afghanistan']
    original_examples = [ex for ex in examples if ex in original_synthesis.lower()]
    rag_examples = [ex for ex in examples if ex in rag_synthesis.lower()]
    
    print(f"Original: {original_examples}")
    print(f"RAG-enhanced: {rag_examples}")
    
    # Summary
    print("\n\n✅ SUMMARY:")
    print("-" * 60)
    print("The RAG-enhanced synthesis typically:")
    print("- Incorporates more specific examples and historical context")
    print("- Makes broader connections across different time periods")
    print("- Provides more nuanced analysis with additional perspectives")
    print("- Grounds abstract patterns in concrete cases")
    
    # Test with another REM cycle
    print("\n\n" + "="*80)
    print("🎲 Running another REM cycle for comparison...")
    print("="*80)
    
    samples2 = rem_cycle._sample_nodes(current_year=None)
    if samples2 and len(samples2) >= 3:
        question2 = rem_cycle._find_implicit_question(samples2)
        print(f"\nNew Question: {question2}")
        
        # Show node types sampled
        print(f"\nSampled node types: {[s.metadata.get('node_type', 'unknown') for s in samples2]}")
        
        # Quick RAG lookup
        rag_results2 = vector_store.query(question2, k=5)
        print(f"RAG found {len(rag_results2['documents'])} additional relevant passages")
        
        # Show variety of node types in RAG results
        rag_types = [m.get('node_type', 'unknown') for m in rag_results2['metadatas']]
        print(f"RAG node types: {rag_types}")


if __name__ == "__main__":
    test_rem_rag_comparison()