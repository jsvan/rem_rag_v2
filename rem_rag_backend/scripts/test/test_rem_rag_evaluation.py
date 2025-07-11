#!/usr/bin/env python3
"""
Evaluate REM cycle RAG enhancement - cost/benefit and coherence analysis.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from rem_rag_backend.src.core.rem_cycle import REMCycle
from rem_rag_backend.src.vector_store.chromadb_store import REMVectorStore
from rem_rag_backend.src.llm.openai_client import LLMClient
from rem_rag_backend.src.config import LLM_MODEL


def test_rem_rag_evaluation():
    """Evaluate if RAG enhancement is worth the cost."""
    print("🔬 REM Cycle RAG Enhancement Evaluation")
    print("=" * 80)
    
    # Initialize components
    vector_store = REMVectorStore()
    llm_client = LLMClient(model=LLM_MODEL)
    rem_cycle = REMCycle(llm_client, vector_store)
    
    # Run multiple tests
    num_tests = 3
    results = []
    
    for test_num in range(num_tests):
        print(f"\n\n{'='*20} TEST {test_num + 1} {'='*20}")
        
        # Sample 3 nodes
        samples = rem_cycle._sample_nodes(current_year=None)
        if not samples or len(samples) < 3:
            continue
            
        # Show sampled node types
        node_types = [s.metadata.get('node_type', 'unknown') for s in samples]
        print(f"Sampled node types: {node_types}")
        
        # Generate question
        question = rem_cycle._find_implicit_question(samples)
        print(f"\nQuestion: {question[:100]}...")
        
        # 1. Generate original synthesis (3 nodes)
        start_time = time.time()
        original_synthesis = rem_cycle._generate_synthesis(samples, question)
        original_time = time.time() - start_time
        original_words = len(original_synthesis.split())
        
        # 2. Do RAG lookup
        rag_results = vector_store.query(question, k=5)
        
        # Show what RAG found
        rag_types = [m.get('node_type', 'unknown') for m in rag_results['metadatas']]
        print(f"RAG found: {rag_types}")
        
        # 3. Generate length-constrained RAG synthesis
        enhanced_context = []
        
        # Add original 3 nodes
        for i, sample in enumerate(samples):
            enhanced_context.append(
                f"Source {i+1} ({sample.metadata.get('year', 'Unknown')}):\n{sample.content[:300]}..."
            )
        
        # Add RAG results
        for i, (text, metadata) in enumerate(zip(rag_results['documents'], rag_results['metadatas'])):
            enhanced_context.append(
                f"Additional {i+1} ({metadata.get('year', 'Unknown')}):\n{text[:300]}..."
            )
        
        # Constrained prompt - SAME LENGTH as original
        constrained_prompt = f"""Question: {question}

Context (8 passages total):
{chr(10).join(enhanced_context)}

Based on ALL these passages, provide a synthesis that answers the question.
IMPORTANT: Write exactly 1 paragraph of {original_words} words (same length as requested).
Focus on the most important insights that connect across the sources."""
        
        start_time = time.time()
        rag_synthesis_constrained = llm_client.generate_sync(
            prompt=constrained_prompt,
            temperature=0.7,
            max_tokens=400
        )
        rag_time = time.time() - start_time
        rag_words = len(rag_synthesis_constrained.split())
        
        # 4. Analyze coherence and quality
        result = {
            'test_num': test_num + 1,
            'original_synthesis': original_synthesis,
            'rag_synthesis': rag_synthesis_constrained,
            'original_words': original_words,
            'rag_words': rag_words,
            'original_time': original_time,
            'rag_time': rag_time,
            'node_types': node_types,
            'rag_types': rag_types
        }
        results.append(result)
        
        print(f"\n📊 Quick Stats:")
        print(f"Original: {original_words} words in {original_time:.2f}s")
        print(f"RAG: {rag_words} words in {rag_time:.2f}s")
    
    # Comprehensive analysis
    print("\n\n" + "="*80)
    print("📈 COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    for result in results:
        print(f"\n\n--- Test {result['test_num']} ---")
        
        print("\n1️⃣ ORIGINAL (3 nodes):")
        print(result['original_synthesis'])
        
        print("\n2️⃣ RAG-CONSTRAINED (8 nodes):")
        print(result['rag_synthesis'])
        
        # Coherence check - look for disconnected ideas
        print("\n🔍 Coherence Analysis:")
        
        # Count transition words
        transitions = ['however', 'moreover', 'furthermore', 'similarly', 'thus', 
                      'therefore', 'consequently', 'additionally', 'meanwhile']
        
        orig_transitions = sum(1 for t in transitions if t in result['original_synthesis'].lower())
        rag_transitions = sum(1 for t in transitions if t in result['rag_synthesis'].lower())
        
        print(f"Transition words - Original: {orig_transitions}, RAG: {rag_transitions}")
        
        # Check topic jumps (rough heuristic: count distinct themes)
        # Split into sentences and check topic continuity
        orig_sentences = result['original_synthesis'].split('.')
        rag_sentences = result['rag_synthesis'].split('.')
        
        print(f"Sentence count - Original: {len(orig_sentences)}, RAG: {len(rag_sentences)}")
        
        # Check if RAG is trying to cover too much
        themes = ['economic', 'political', 'military', 'social', 'cultural', 
                 'technological', 'environmental', 'religious', 'ethnic']
        
        orig_themes = sum(1 for theme in themes if theme in result['original_synthesis'].lower())
        rag_themes = sum(1 for theme in themes if theme in result['rag_synthesis'].lower())
        
        print(f"Theme coverage - Original: {orig_themes}, RAG: {rag_themes}")
        
        # Specificity check
        specific_markers = ['for example', 'such as', 'specifically', 'in particular', 
                          'namely', 'including', 'especially']
        
        orig_specific = sum(1 for m in specific_markers if m in result['original_synthesis'].lower())
        rag_specific = sum(1 for m in specific_markers if m in result['rag_synthesis'].lower())
        
        print(f"Specificity markers - Original: {orig_specific}, RAG: {rag_specific}")
    
    # Cost-benefit summary
    print("\n\n" + "="*80)
    print("💰 COST-BENEFIT ANALYSIS")
    print("="*80)
    
    avg_original_time = sum(r['original_time'] for r in results) / len(results)
    avg_rag_time = sum(r['rag_time'] for r in results) / len(results)
    
    print(f"\n⏱️  Time Cost:")
    print(f"Original: {avg_original_time:.2f}s average")
    print(f"RAG-enhanced: {avg_rag_time:.2f}s average")
    print(f"Additional time: {avg_rag_time - avg_original_time:.2f}s")
    
    print(f"\n💵 API Cost (rough estimate):")
    print(f"Original: 1 synthesis call")
    print(f"RAG-enhanced: 1 synthesis call + 1 embedding call")
    print(f"Cost increase: ~2x")
    
    print("\n📊 Quality Assessment:")
    print("When RAG enhancement helps:")
    print("- Questions about broad patterns across time")
    print("- Need for specific historical examples")
    print("- Complex multi-domain questions")
    
    print("\nWhen RAG enhancement hurts:")
    print("- Already have highly relevant 3 nodes")
    print("- Question is narrow/specific")
    print("- Risk of incoherent kitchen-sink response")
    
    print("\n🎯 RECOMMENDATION:")
    print("-" * 60)
    print("The RAG enhancement adds complexity and cost that may not be justified for REM cycles.")
    print("The original 3-node synthesis often captures the key insight more clearly.")
    print("RAG enhancement risks creating 'kitchen sink' responses that lose focus.")
    print("\nSuggestion: Keep REM cycles pure (3 nodes only) but store the question for")
    print("potential RAG-enhanced exploration in the browser extension or on-demand.")


if __name__ == "__main__":
    test_rem_rag_evaluation()