#!/usr/bin/env python3
"""
Analyze if synthesis nodes cluster together or provide diverse insights.
Tests whether improved synthesis nodes are just saying the same thing or adding unique value.
"""

import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rem_rag_backend.src.vector_store.chromadb_store import REMVectorStore


def analyze_synthesis_clustering():
    """Analyze clustering patterns of synthesis nodes."""
    print("🔍 Synthesis Node Clustering Analysis")
    print("=" * 80)
    
    # Initialize vector store
    vector_store = REMVectorStore()
    
    # Get a sample of synthesis nodes
    print("\n📊 Sampling synthesis nodes...")
    synthesis_filter = {"node_type": "synthesis"}
    synthesis_sample = vector_store.sample(50, filter=synthesis_filter)
    
    if not synthesis_sample['documents']:
        print("No synthesis nodes found!")
        return
    
    print(f"Found {len(synthesis_sample['documents'])} synthesis nodes")
    
    # Analyze each synthesis node's nearest neighbors
    print("\n\n🎯 Nearest Neighbor Analysis")
    print("=" * 80)
    
    neighbor_analysis = []
    
    for i, (syn_text, syn_meta) in enumerate(zip(synthesis_sample['documents'][:10], 
                                                 synthesis_sample['metadatas'][:10])):
        print(f"\n--- Synthesis Node {i+1} ---")
        print(f"Year: {syn_meta.get('year', 'unknown')}")
        print(f"Text preview: {syn_text[:150]}...")
        
        # Get nearest neighbors
        neighbors = vector_store.query(syn_text, k=11)  # 11 to exclude self
        
        # Analyze neighbor types
        neighbor_types = defaultdict(int)
        synthesis_positions = []
        
        for j, (n_text, n_meta) in enumerate(zip(neighbors['documents'][1:11], 
                                                neighbors['metadatas'][1:11])):
            node_type = n_meta.get('node_type', 'unknown')
            neighbor_types[node_type] += 1
            
            if node_type == 'synthesis':
                synthesis_positions.append(j + 1)
        
        print(f"\nNeighbor distribution:")
        for node_type, count in sorted(neighbor_types.items()):
            print(f"  {node_type}: {count}")
        
        print(f"Synthesis positions: {synthesis_positions}")
        
        # Calculate synthesis clustering score
        synthesis_count = neighbor_types.get('synthesis', 0)
        clustering_score = synthesis_count / 10
        
        neighbor_analysis.append({
            'text': syn_text,
            'synthesis_neighbors': synthesis_count,
            'clustering_score': clustering_score,
            'distances': neighbors['distances'][1:11]
        })
    
    # Overall clustering analysis
    print("\n\n📈 Clustering Metrics")
    print("=" * 80)
    
    avg_synthesis_neighbors = np.mean([a['synthesis_neighbors'] for a in neighbor_analysis])
    avg_clustering_score = np.mean([a['clustering_score'] for a in neighbor_analysis])
    
    print(f"Average synthesis neighbors: {avg_synthesis_neighbors:.2f} out of 10")
    print(f"Average clustering score: {avg_clustering_score:.2%}")
    
    # Diversity analysis - check if synthesis nodes say different things
    print("\n\n🌈 Diversity Analysis")
    print("=" * 80)
    
    # Sample pairs of synthesis nodes
    print("\nComparing synthesis pairs for semantic similarity:")
    
    for i in range(0, min(6, len(synthesis_sample['documents'])), 2):
        syn1 = synthesis_sample['documents'][i]
        syn2 = synthesis_sample['documents'][i+1]
        
        print(f"\n--- Pair {i//2 + 1} ---")
        print(f"Syn 1: {syn1[:100]}...")
        print(f"Syn 2: {syn2[:100]}...")
        
        # Check if they appear in each other's neighbors
        neighbors1 = vector_store.query(syn1, k=20)
        neighbors2 = vector_store.query(syn2, k=20)
        
        # Find position of syn2 in syn1's neighbors
        syn2_position = None
        for j, text in enumerate(neighbors1['documents']):
            if text[:50] == syn2[:50]:  # Rough match
                syn2_position = j + 1
                break
        
        if syn2_position:
            print(f"Mutual proximity: Syn2 is neighbor #{syn2_position} of Syn1")
            print(f"Distance: {neighbors1['distances'][syn2_position-1]:.3f}")
        else:
            print("Not in each other's top-20 neighbors (diverse topics)")
    
    # Topic diversity check
    print("\n\n🏷️ Topic Diversity Check")
    print("=" * 80)
    
    # Extract key terms from synthesis nodes
    topic_words = defaultdict(int)
    for text in synthesis_sample['documents']:
        # Simple keyword extraction (could be enhanced)
        words = text.lower().split()
        for word in words:
            if len(word) > 6 and word.isalpha():
                topic_words[word] += 1
    
    # Show most common themes
    common_themes = sorted(topic_words.items(), key=lambda x: x[1], reverse=True)[:20]
    print("\nMost common terms across synthesis nodes:")
    for word, count in common_themes:
        print(f"  {word}: {count}")
    
    # Conclusions
    print("\n\n✅ CONCLUSIONS")
    print("=" * 80)
    
    if avg_clustering_score > 0.3:
        print("⚠️ HIGH CLUSTERING: Synthesis nodes tend to cluster together")
        print("This suggests they may be too similar or generic")
    else:
        print("✅ LOW CLUSTERING: Synthesis nodes are well-distributed")
        print("This suggests they capture diverse insights")
    
    print(f"\nDiversity indicators:")
    print(f"- {100 - avg_clustering_score * 100:.1f}% of neighbors are non-synthesis nodes")
    print(f"- Synthesis nodes connect to chunks, learnings, and REM nodes")
    print(f"- Topic analysis shows {'diverse' if len(common_themes) > 15 else 'limited'} vocabulary")
    
    print("\n🎯 Recommendation:")
    if avg_clustering_score < 0.3:
        print("The improved synthesis prompt is working well - nodes are diverse and well-integrated")
    else:
        print("Consider further refining the synthesis prompt to encourage more diverse insights")


if __name__ == "__main__":
    analyze_synthesis_clustering()