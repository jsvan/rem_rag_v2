#!/usr/bin/env python3
"""
Query random nodes from the vector store with various options.

Usage:
    python query_nodes.py --node-type chunk --neighbors --count
    python query_nodes.py --node-type rem --neighbors
    python query_nodes.py --node-type synthesis
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from rem_rag_backend.src.vector_store.chromadb_store import REMVectorStore
from rem_rag_backend.src.config import NODE_TYPES


def format_node(node: dict, index: int = 0) -> str:
    """Format a node for display."""
    metadata = node.get('metadata', {})
    text = node.get('text', '')
    
    output = f"\n{'='*80}\n"
    output += f"Node {index + 1}:\n"
    output += f"Type: {metadata.get('node_type', 'unknown')}\n"
    output += f"Year: {metadata.get('year', 'unknown')}\n"
    
    if 'title' in metadata:
        output += f"Title: {metadata['title']}\n"
    if 'entity' in metadata:
        output += f"Entity: {metadata['entity']}\n"
    if 'generation' in metadata:
        output += f"Generation: {metadata['generation']}\n"
    
    output += f"\nText ({len(text)} chars):\n"
    output += "-" * 40 + "\n"
    output += text
    output += "\n"
    
    return output


def query_nodes(
    node_type: Optional[str] = None,
    get_neighbors: bool = False,
    count_only: bool = False,
    num_samples: int = 1
) -> None:
    """Query nodes from the vector store."""
    
    # Initialize vector store
    vector_store = REMVectorStore()
    
    # If count only, get counts for all types or specific type
    if count_only:
        if node_type:
            filter_dict = {"node_type": node_type}
            results = vector_store.query("", k=1, filter=filter_dict)
            # Get total count from collection
            collection = vector_store.collection
            count = collection.count()
            
            # Count with filter
            all_docs = collection.get(where=filter_dict)
            filtered_count = len(all_docs['ids']) if all_docs['ids'] else 0
            
            print(f"\nNode type '{node_type}': {filtered_count} nodes")
        else:
            # Get counts for all node types
            print("\nNode counts by type:")
            print("-" * 40)
            
            total = 0
            for nt in NODE_TYPES:
                filter_dict = {"node_type": nt}
                all_docs = vector_store.collection.get(where=filter_dict)
                count = len(all_docs['ids']) if all_docs['ids'] else 0
                total += count
                print(f"{nt:20} {count:6}")
            
            print("-" * 40)
            print(f"{'Total:':20} {total:6}")
        
        return
    
    # Sample random nodes
    filter_dict = {"node_type": node_type} if node_type else None
    
    try:
        sampled = vector_store.sample(n=num_samples, filter=filter_dict)
        
        if not sampled or not sampled.get('documents'):
            print(f"No nodes found with type '{node_type}'" if node_type else "No nodes found")
            return
        
        documents = sampled['documents']
        metadatas = sampled['metadatas']
        ids = sampled['ids']
        
        print(f"\nSampled {len(documents)} random node(s)" + 
              (f" of type '{node_type}'" if node_type else ""))
        
        for i, (text, metadata, node_id) in enumerate(zip(documents, metadatas, ids)):
            node = {'text': text, 'metadata': metadata, 'id': node_id}
            print(format_node(node, i))
            
            if get_neighbors:
                # Get neighbors - don't filter by node type for neighbors
                results = vector_store.query(text, k=5)
                
                if results and results.get('documents'):
                    neighbor_docs = results['documents']
                    neighbor_metas = results['metadatas']
                    neighbor_ids = results['ids']
                    neighbor_dists = results.get('distances', [])
                    
                    print(f"\n{len(neighbor_docs)} Nearest Neighbors:")
                    for j, (ndoc, nmeta, nid, ndist) in enumerate(zip(neighbor_docs, neighbor_metas, neighbor_ids, neighbor_dists or [None]*len(neighbor_docs))):
                        # Skip if it's the same node
                        if nid != node_id:
                            distance_str = f"{ndist:.3f}" if ndist is not None else "N/A"
                            print(f"\n  Neighbor {j + 1} (distance: {distance_str}):")
                            print(f"  Type: {nmeta.get('node_type', 'unknown')}")
                            print(f"  Year: {nmeta.get('year', 'unknown')}")
                            print(f"  Full text: {ndoc}")
    
    except Exception as e:
        print(f"Error querying nodes: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Query random nodes from the REM RAG vector store"
    )
    
    parser.add_argument(
        "--node-type",
        choices=NODE_TYPES,
        help="Filter by node type"
    )
    
    parser.add_argument(
        "--neighbors",
        action="store_true",
        help="Also retrieve 5 nearest neighbors"
    )
    
    parser.add_argument(
        "--count",
        action="store_true",
        help="Only return counts, not content"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of random nodes to sample (default: 1)"
    )
    
    args = parser.parse_args()
    
    query_nodes(
        node_type=args.node_type,
        get_neighbors=args.neighbors,
        count_only=args.count,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()