#!/usr/bin/env python3
"""
Quick test script to verify the research module setup.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Core modules
        from research.core import EmbeddingsLoader, VectorSpaceMetrics, VectorSpaceAnalyzer
        print("✓ Core modules imported successfully")
        
        # Utils
        from research.utils import ConvexHullAnalysis
        from research.utils import statistics
        print("✓ Utils modules imported successfully")
        
        # Visualizations
        from research.visualizations import density_plots, coverage_plots, projections
        print("✓ Visualization modules imported successfully")
        
        # REM RAG modules
        from src.vector_store import REMVectorStore
        from src.config import COLLECTION_NAME
        print("✓ REM RAG modules imported successfully")
        
        # External dependencies
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.decomposition import PCA
        print("✓ External dependencies imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_vector_store_connection():
    """Test connection to vector store."""
    print("\nTesting vector store connection...")
    
    try:
        from src.vector_store import REMVectorStore
        from src.config import COLLECTION_NAME
        
        store = REMVectorStore(collection_name=COLLECTION_NAME)
        stats = store.get_stats()
        
        print(f"✓ Connected to collection: {COLLECTION_NAME}")
        print(f"  Total documents: {stats['total_documents']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Vector store error: {e}")
        return False


def test_basic_functionality():
    """Test basic analysis functionality."""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        from research.core.metrics import VectorSpaceMetrics
        from research.utils.geometry import ConvexHullAnalysis
        
        # Create some test embeddings
        embeddings = np.random.randn(100, 10)
        
        # Test metrics
        metrics = VectorSpaceMetrics()
        diversity = metrics.calculate_diversity_index(embeddings)
        print(f"✓ Calculated diversity index: {diversity:.4f}")
        
        # Test convex hull
        hull_analyzer = ConvexHullAnalysis(n_components=3)
        volume = hull_analyzer.calculate_hull_volume(embeddings)
        print(f"✓ Calculated hull volume: {volume:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*50)
    print("Vector Space Analysis Module Test")
    print("="*50)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_vector_store_connection()
    all_passed &= test_basic_functionality()
    
    print("\n" + "="*50)
    if all_passed:
        print("✅ All tests passed! The module is ready to use.")
        print("\nTo run the full analysis:")
        print("  python research/scripts/run_analysis.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("="*50)


if __name__ == "__main__":
    main()