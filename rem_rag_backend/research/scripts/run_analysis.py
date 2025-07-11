#!/usr/bin/env python3
"""
Main script to run the vector space coverage analysis.

Usage: 
    python research/scripts/run_analysis.py
    python research/scripts/run_analysis.py --sample-size 1000
    python research/scripts/run_analysis.py --visualize-only
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import our modules
from research.core.analysis import VectorSpaceAnalyzer
from research.core.embeddings_loader import EmbeddingsLoader
from research.core.metrics import VectorSpaceMetrics
from research.visualizations import density_plots, coverage_plots, projections

# Import REM RAG modules
from src.vector_store import REMVectorStore
from src.config import COLLECTION_NAME

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_visualizations(analyzer: VectorSpaceAnalyzer, output_dir: Path):
    """Create all visualizations from the analysis results."""
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    # Get data
    embeddings_data = analyzer.embeddings_data
    analysis_results = analyzer.analysis_results
    
    if not embeddings_data:
        logger.warning("No embeddings data available for visualization")
        return
    
    # Extract embeddings and metadata
    embeddings_dict = {
        node_type: data['embeddings'] 
        for node_type, data in embeddings_data.items()
    }
    
    metadata_dict = {
        node_type: data['metadata']
        for node_type, data in embeddings_data.items()
    }
    
    # 1. Density plots
    print("  - Creating density comparison...")
    density_plots.plot_density_comparison(
        embeddings_dict, 
        plots_dir / "density_comparison.png"
    )
    
    print("  - Creating distance distributions...")
    density_plots.plot_distance_distributions(
        embeddings_dict,
        plots_dir / "distance_distributions.png"
    )
    
    # Create individual density heatmaps
    for node_type, embeddings in embeddings_dict.items():
        if len(embeddings) >= 10:
            density_plots.create_density_heatmap(
                embeddings,
                node_type,
                plots_dir / f"density_heatmap_{node_type}.png"
            )
    
    # Local density comparison
    if 'node_metrics' in analysis_results:
        local_densities_dict = {}
        for node_type, metrics in analysis_results['node_metrics'].items():
            if node_type in embeddings_dict and 'mean_local_density' in metrics:
                # Calculate local densities for this type
                metrics_calc = VectorSpaceMetrics()
                densities = metrics_calc.calculate_local_density(embeddings_dict[node_type])
                local_densities_dict[node_type] = densities
        
        if local_densities_dict:
            print("  - Creating local density comparison...")
            density_plots.plot_local_density_comparison(
                embeddings_dict,
                local_densities_dict,
                plots_dir / "local_density_comparison.png"
            )
    
    print("  - Creating density summary...")
    density_plots.create_density_summary_plot(
        analysis_results,
        plots_dir / "density_summary.png"
    )
    
    # 2. Coverage plots
    print("  - Creating convex hull visualization...")
    coverage_plots.plot_convex_hulls_2d(
        embeddings_dict,
        plots_dir / "convex_hulls_2d.png"
    )
    
    print("  - Creating hull metrics comparison...")
    coverage_plots.plot_hull_metrics_comparison(
        analysis_results,
        plots_dir / "hull_metrics.png"
    )
    
    print("  - Creating overlap matrix...")
    coverage_plots.plot_coverage_overlap_matrix(
        analysis_results,
        plots_dir / "overlap_matrix.png"
    )
    
    # Coverage gaps visualization
    if 'coverage_gaps' in analysis_results:
        gaps = analysis_results['coverage_gaps']
        if 'empty_region_centers' in gaps and gaps['empty_region_centers']:
            print("  - Creating coverage gaps visualization...")
            import numpy as np
            gap_centers = np.array(gaps['empty_region_centers'])
            coverage_plots.plot_coverage_gaps_visualization(
                embeddings_dict,
                gap_centers,
                plots_dir / "coverage_gaps.png"
            )
    
    print("  - Creating coverage summary figure...")
    coverage_plots.create_coverage_summary_figure(
        analysis_results,
        plots_dir / "coverage_summary.png"
    )
    
    # 3. Projection plots
    print("  - Creating PCA projections...")
    projections.plot_pca_projections(
        embeddings_dict,
        plots_dir / "pca_analysis.png"
    )
    
    print("  - Creating t-SNE projection...")
    projections.plot_tsne_projection(
        embeddings_dict,
        plots_dir / "tsne_projection.png"
    )
    
    print("  - Creating UMAP projection...")
    projections.plot_umap_projection(
        embeddings_dict,
        plots_dir / "umap_projection.png"
    )
    
    print("  - Creating temporal evolution plot...")
    projections.plot_temporal_evolution(
        embeddings_dict,
        metadata_dict,
        plots_dir / "temporal_evolution.png"
    )
    
    print("  - Creating projection comparison...")
    projections.create_projection_comparison(
        embeddings_dict,
        plots_dir / "projection_comparison.png"
    )
    
    print(f"\nAll visualizations saved to: {plots_dir}")


def main():
    """Main entry point for the analysis."""
    
    parser = argparse.ArgumentParser(
        description="Analyze vector space coverage in REM RAG"
    )
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=None,
        help="Limit number of embeddings to analyze"
    )
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only create visualizations from existing results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="research/outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=COLLECTION_NAME,
        help="ChromaDB collection name"
    )
    
    args = parser.parse_args()
    
    # Initialize vector store
    print(f"Connecting to ChromaDB collection: {args.collection}")
    vector_store = REMVectorStore(collection_name=args.collection)
    
    # Get initial statistics
    stats = vector_store.get_stats()
    print(f"Total documents in collection: {stats['total_documents']}")
    
    # Create analyzer
    analyzer = VectorSpaceAnalyzer(vector_store, output_dir=args.output_dir)
    
    if not args.visualize_only:
        # Run full analysis
        print("\nStarting vector space analysis...")
        results = analyzer.run_full_analysis(sample_size=args.sample_size)
        
        # Print quick summary
        print(analyzer.quick_summary())
    else:
        # Load existing results if available
        print("\nLoading embeddings for visualization...")
        analyzer.embeddings_data = analyzer.loader.load_all_embeddings(
            limit=args.sample_size
        )
        
        if not analyzer.embeddings_data:
            print("No embeddings found. Run full analysis first.")
            return
    
    # Create visualizations
    create_visualizations(analyzer, Path(args.output_dir))
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    # Print key recommendations
    if 'recommendations' in analyzer.analysis_results:
        print("\nKey Recommendations:")
        for i, rec in enumerate(analyzer.analysis_results['recommendations'][:5]):
            print(f"\n{i+1}. [{rec['severity'].upper()}] {rec['message']}")
            if 'suggestion' in rec:
                print(f"   → {rec['suggestion']}")


if __name__ == "__main__":
    main()