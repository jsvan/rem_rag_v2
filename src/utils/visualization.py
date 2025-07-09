"""Simple utilities for visualizing results"""


def display_entity_evolution(evolution: list) -> None:
    """
    Display entity evolution in a readable format.
    
    Args:
        evolution: List of nodes tracking entity over time
    """
    if not evolution:
        print("No evolution data available")
        return
        
    for node in evolution:
        print(f"\n{'='*60}")
        print(f"Year: {node.get('year', 'Unknown')}")
        print(f"Type: {node.get('node_type', 'Unknown')}")
        print(f"Depth: {node.get('generation_depth', 0)}")
        print(f"ID: {node.get('id', 'Unknown')}")
        print(f"\nContent:")
        print(f"{node.get('text', '')[:500]}...")


def display_rem_insights(insights: list) -> None:
    """
    Display REM insights in a readable format.
    
    Args:
        insights: List of REM insight documents
    """
    if not insights:
        print("No REM insights available")
        return
        
    for i, (text, meta) in enumerate(insights):
        print(f"\n{'='*60}")
        print(f"REM Insight {i+1}")
        print(f"Year: {meta.get('year', 'Unknown')}")
        print(f"Source Nodes: {meta.get('source_node_ids', [])[:3]}")
        print(f"\nInsight:")
        print(text)


def display_search_results(results: dict, limit: int = 5) -> None:
    """
    Display search results in a readable format.
    
    Args:
        results: Search results dict with documents, metadatas, ids
        limit: Maximum number of results to display
    """
    if not results.get('documents'):
        print("No results found")
        return
        
    docs = results['documents'][:limit]
    metas = results['metadatas'][:limit]
    ids = results['ids'][:limit]
    
    for i, (doc, meta, doc_id) in enumerate(zip(docs, metas, ids)):
        print(f"\n{'='*60}")
        print(f"Result {i+1}")
        print(f"ID: {doc_id}")
        print(f"Type: {meta.get('node_type', 'Unknown')}")
        print(f"Year: {meta.get('year', 'Unknown')}")
        print(f"Article: {meta.get('article_title', 'Unknown')}")
        print(f"\nContent:")
        print(f"{doc[:300]}...")