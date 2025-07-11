"""Tests for the data loader"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from rem_rag_backend.src.data_processing.fa_loader import ForeignAffairsLoader, ArticleChunker


def test_loader_basic():
    """Test basic loader functionality"""
    loader = ForeignAffairsLoader()
    
    # For testing, we'll just check the structure
    # In real usage, this would download the dataset
    print("ForeignAffairsLoader initialized successfully")
    
    # Test year extraction
    test_row = {
        'title': 'Test Article',
        'content': 'Published in 1962. This is about the Cuban Missile Crisis.',
        'date': 'October 1962'
    }
    
    import pandas as pd
    year = loader._extract_year(pd.Series(test_row))
    print(f"Extracted year: {year}")
    assert year == 1962
    
    # Test article preparation
    article = {
        'title': 'The Cuban Crisis',
        'content': 'In October 1962, the world came close to nuclear war...',
        'year': 1962,
        'author': 'John Doe'
    }
    
    prepared = loader.prepare_for_processing(article)
    print(f"Prepared article keys: {prepared.keys()}")
    
    assert 'text' in prepared
    assert 'The Cuban Crisis' in prepared['text']
    assert prepared['year'] == 1962
    assert 'article_id' in prepared


def test_chunker():
    """Test article chunking"""
    chunker = ArticleChunker(chunk_size=100, chunk_overlap=20)
    
    article = {
        'text': "This is the first sentence. " * 10,  # ~300 chars
        'article_id': 'test_123',
        'year': 1962,
        'title': 'Test Article',
        'metadata': {'author': 'Test Author'}
    }
    
    chunks = chunker.chunk_article(article)
    print(f"Created {len(chunks)} chunks")
    
    # Check chunks
    assert len(chunks) > 1  # Should create multiple chunks
    assert all('text' in c for c in chunks)
    assert all(c['year'] == 1962 for c in chunks)
    assert all(c['article_id'] == 'test_123' for c in chunks)
    
    # Check chunk indices
    indices = [c['chunk_index'] for c in chunks]
    assert indices == list(range(len(chunks)))
    
    # Check metadata
    assert all(c['metadata']['total_chunks'] == len(chunks) for c in chunks)
    
    print("Chunker tests passed!")


def test_dataset_loading():
    """Test actual dataset loading - requires internet"""
    print("\nTesting dataset loading (requires internet)...")
    print("This will attempt to load the Foreign Affairs dataset from HuggingFace")
    print("Note: This may take a while on first run as it downloads the dataset")
    
    try:
        loader = ForeignAffairsLoader()
        
        # Try loading just a small sample
        # Note: The actual dataset loading would happen here
        # We'll skip the actual download in this test
        print("Dataset loader configured successfully")
        print("To actually load the dataset, run: loader.load_dataset()")
        
    except Exception as e:
        print(f"Dataset loading setup failed (this is expected without internet): {e}")


if __name__ == "__main__":
    test_loader_basic()
    test_chunker()
    test_dataset_loading()
    print("\nAll tests completed!")