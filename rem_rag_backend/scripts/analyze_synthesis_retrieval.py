#!/usr/bin/env python3
"""
Analyze how synthesis nodes are retrieved compared to other node types.
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from rem_rag_backend.src.vector_store.chromadb_store import REMVectorStore


def analyze_synthesis_retrieval():
    """Analyze synthesis node retrieval patterns."""
    print("🔍 Analyzing Synthesis Node Retrieval Patterns")
    print("=" * 80)
    
    # Initialize vector store
    vector_store = REMVectorStore()
    
    # Test queries
    test_queries = [
        "What are the patterns of humanitarian intervention?",
        "How has nuclear deterrence evolved?",
        "What role does sovereignty play in international relations?",
        "What are the consequences of economic sanctions?",
        "How do great powers maintain influence?"
    ]
    
    # For each query, analyze node type distribution
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        # Get top 30 results
        results = vector_store.query(query, k=30)
        
        # Count node types
        node_type_counts = defaultdict(int)
        synthesis_positions = []
        
        for i, metadata in enumerate(results['metadatas']):
            node_type = metadata.get('node_type', 'unknown')
            node_type_counts[node_type] += 1
            
            if node_type == 'synthesis':
                synthesis_positions.append(i + 1)  # 1-based position
        
        # Print distribution
        print("Node type distribution in top 30:")
        for node_type, count in sorted(node_type_counts.items()):
            percentage = (count / 30) * 100
            print(f"  {node_type:15}: {count:3} ({percentage:5.1f}%)")
        
        if synthesis_positions:
            print(f"Synthesis node positions: {synthesis_positions}")
            avg_position = sum(synthesis_positions) / len(synthesis_positions)
            print(f"Average synthesis position: {avg_position:.1f}")
    
    print("\n" + "=" * 80)
    print("📊 Overall Analysis")
    print("=" * 80)
    
    # Get some example synthesis nodes
    synthesis_filter = {"node_type": "synthesis"}
    synthesis_examples = vector_store.sample(5, filter=synthesis_filter)
    
    print("\nExample synthesis nodes:")
    for i, (text, metadata) in enumerate(zip(synthesis_examples['documents'], synthesis_examples['metadatas'])):
        print(f"\n--- Synthesis Example {i+1} ---")
        print(f"Year: {metadata.get('year', 'unknown')}")
        print(f"Entity: {metadata.get('entity', 'N/A')}")
        print(f"Text preview: {text[:200]}...")
    
    # Compare similarity scores
    print("\n🔬 Similarity Score Analysis")
    print("-" * 60)
    
    # Query for synthesis nodes specifically
    test_query = "humanitarian intervention"
    
    # Get synthesis results
    synthesis_results = vector_store.query(test_query, k=10, filter={"node_type": "synthesis"})
    
    # Get all results
    all_results = vector_store.query(test_query, k=30)
    
    print(f"\nQuery: '{test_query}'")
    print(f"Best synthesis node distance: {synthesis_results['distances'][0]:.3f}")
    print(f"Best overall node distance: {all_results['distances'][0]:.3f}")
    
    # Find where synthesis nodes appear in overall results
    synthesis_distances = []
    for i, metadata in enumerate(all_results['metadatas']):
        if metadata.get('node_type') == 'synthesis':
            synthesis_distances.append((i+1, all_results['distances'][i]))
    
    if synthesis_distances:
        print("\nSynthesis nodes in top 30:")
        for pos, dist in synthesis_distances[:5]:  # Show first 5
            print(f"  Position {pos}: distance = {dist:.3f}")


if __name__ == "__main__":
    analyze_synthesis_retrieval()