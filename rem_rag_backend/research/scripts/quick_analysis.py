#!/usr/bin/env python3
"""
Quick analysis to get immediate insights about the vector space.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.vector_store import REMVectorStore
from research.core.embeddings_loader import EmbeddingsLoader
from research.core.metrics import VectorSpaceMetrics
import numpy as np


def main():
    print("Running quick vector space analysis...")
    print("="*50)
    
    # Initialize vector store
    store = REMVectorStore()
    loader = EmbeddingsLoader(store)
    metrics = VectorSpaceMetrics()
    
    # Get sample data for each node type
    print("\n1. Sampling embeddings by node type...")
    sample_data = loader.get_sample_embeddings(n_samples=100)
    
    print("\n2. Node type distribution:")
    for node_type, data in sample_data.items():
        count = len(data['embeddings'])
        print(f"   - {node_type}: {count} samples")
    
    print("\n3. Quick metrics for each node type:")
    for node_type, data in sample_data.items():
        embeddings = data['embeddings']
        if len(embeddings) > 1:
            # Calculate key metrics
            diversity = metrics.calculate_diversity_index(embeddings)
            local_densities = metrics.calculate_local_density(embeddings, k=min(10, len(embeddings)-1))
            mean_density = np.mean(local_densities)
            
            print(f"\n   {node_type}:")
            print(f"     - Diversity index: {diversity:.4f}")
            print(f"     - Mean local density: {mean_density:.4f}")
            print(f"     - Embedding dimension: {embeddings.shape[1]}")
    
    # Compare chunk vs synthesis
    if 'chunk' in sample_data and 'synthesis' in sample_data:
        print("\n4. Chunk vs Synthesis comparison:")
        comparison = metrics.compare_distributions(
            sample_data['chunk']['embeddings'],
            sample_data['synthesis']['embeddings']
        )
        
        chunk_to_synth = comparison.get('mean_nn_distance_1to2', 0)
        synth_to_chunk = comparison.get('mean_nn_distance_2to1', 0)
        
        print(f"   - Mean distance from chunks to nearest synthesis: {chunk_to_synth:.4f}")
        print(f"   - Mean distance from synthesis to nearest chunk: {synth_to_chunk:.4f}")
        
        if chunk_to_synth < 0.3:
            print("   ⚠️  Synthesis nodes are very close to chunks - low coverage expansion")
        elif chunk_to_synth > 0.7:
            print("   ✅ Synthesis nodes are exploring new regions!")
    
    # Get total counts
    print("\n5. Total counts in database:")
    stats = store.get_stats()
    print(f"   - Total documents: {stats['total_documents']}")
    
    print("\n" + "="*50)
    print("Quick analysis complete!")
    print("\nFor full analysis with visualizations, run:")
    print("  python research/scripts/run_analysis.py")


if __name__ == "__main__":
    main()