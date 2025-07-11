"""
Main analysis orchestrator for vector space coverage analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
from pathlib import Path
import logging

from .embeddings_loader import EmbeddingsLoader
from .metrics import VectorSpaceMetrics
from ..utils.geometry import ConvexHullAnalysis

logger = logging.getLogger(__name__)


class VectorSpaceAnalyzer:
    """Orchestrates the complete vector space analysis."""
    
    def __init__(self, vector_store, output_dir: str = "research/outputs"):
        """
        Initialize analyzer with vector store.
        
        Args:
            vector_store: REMVectorStore instance
            output_dir: Directory for saving outputs
        """
        self.vector_store = vector_store
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.loader = EmbeddingsLoader(vector_store)
        self.metrics = VectorSpaceMetrics()
        self.hull_analyzer = ConvexHullAnalysis()
        
        # Results storage
        self.embeddings_data = None
        self.analysis_results = {}
        
    def run_full_analysis(self, sample_size: Optional[int] = None) -> Dict:
        """
        Run complete analysis pipeline.
        
        Args:
            sample_size: Optional limit on embeddings to analyze
            
        Returns:
            Complete analysis results
        """
        print("\n" + "="*60)
        print("VECTOR SPACE COVERAGE ANALYSIS")
        print("="*60)
        
        # 1. Load embeddings
        print("\n1. Loading embeddings...")
        self.embeddings_data = self.loader.load_all_embeddings(limit=sample_size)
        
        if not self.embeddings_data:
            logger.error("No embeddings found in database")
            return {'error': 'No embeddings found'}
        
        # Print summary
        print(f"\nLoaded embeddings for {len(self.embeddings_data)} node types:")
        for node_type, data in self.embeddings_data.items():
            print(f"  - {node_type}: {len(data['embeddings'])} embeddings")
        
        # 2. Calculate metrics for each type
        print("\n2. Calculating metrics for each node type...")
        self.analysis_results['node_metrics'] = {}
        
        for node_type, data in self.embeddings_data.items():
            print(f"\n  Analyzing {node_type}...")
            embeddings = data['embeddings']
            
            # Basic metrics
            metrics = self.metrics.calculate_all_metrics(embeddings)
            
            # Hull properties
            hull_props = self.hull_analyzer.analyze_hull_properties(embeddings)
            metrics.update(hull_props)
            
            self.analysis_results['node_metrics'][node_type] = metrics
        
        # 3. Compare coverage between types
        print("\n3. Comparing coverage between node types...")
        self.analysis_results['comparisons'] = self._compare_node_types()
        
        # 4. Identify gaps
        print("\n4. Identifying coverage gaps...")
        self.analysis_results['coverage_gaps'] = self._identify_gaps()
        
        # 5. Generate recommendations
        print("\n5. Generating recommendations...")
        self.analysis_results['recommendations'] = self._generate_recommendations()
        
        # 6. Save results
        self._save_results()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("="*60)
        
        return self.analysis_results
    
    def _compare_node_types(self) -> Dict:
        """Compare coverage between different node types."""
        comparisons = {}
        
        # Compare chunk vs synthetic types
        if 'chunk' in self.embeddings_data:
            chunk_embeddings = self.embeddings_data['chunk']['embeddings']
            
            for node_type in ['synthesis', 'learning', 'rem']:
                if node_type in self.embeddings_data:
                    synthetic_embeddings = self.embeddings_data[node_type]['embeddings']
                    
                    # Hull overlap
                    overlap = self.hull_analyzer.calculate_hull_overlap(
                        chunk_embeddings, synthetic_embeddings
                    )
                    
                    # Distribution comparison
                    dist_comp = self.metrics.compare_distributions(
                        chunk_embeddings, synthetic_embeddings
                    )
                    
                    comparisons[f'chunk_vs_{node_type}'] = {
                        'hull_overlap': overlap,
                        'distribution_comparison': dist_comp
                    }
        
        # Compare synthetic types among themselves
        synthetic_types = ['synthesis', 'learning', 'rem']
        for i, type1 in enumerate(synthetic_types):
            for type2 in synthetic_types[i+1:]:
                if type1 in self.embeddings_data and type2 in self.embeddings_data:
                    overlap = self.hull_analyzer.calculate_hull_overlap(
                        self.embeddings_data[type1]['embeddings'],
                        self.embeddings_data[type2]['embeddings']
                    )
                    comparisons[f'{type1}_vs_{type2}'] = {'hull_overlap': overlap}
        
        return comparisons
    
    def _identify_gaps(self) -> Dict:
        """Identify regions with low coverage."""
        gaps_info = {}
        
        # Combine all embeddings
        all_embeddings = []
        for data in self.embeddings_data.values():
            all_embeddings.append(data['embeddings'])
        
        if all_embeddings:
            all_embeddings = np.vstack(all_embeddings)
            
            # Find empty regions
            empty_regions = self.hull_analyzer.find_empty_regions(all_embeddings)
            
            gaps_info['n_empty_regions'] = len(empty_regions)
            gaps_info['empty_region_centers'] = empty_regions.tolist() if len(empty_regions) > 0 else []
            
            # Calculate coverage by type in empty regions
            if len(empty_regions) > 0:
                coverage_by_type = {}
                for node_type, data in self.embeddings_data.items():
                    # Find nearest neighbor distance from empty regions to this type
                    from sklearn.neighbors import NearestNeighbors
                    nn = NearestNeighbors(n_neighbors=1, metric='cosine')
                    nn.fit(data['embeddings'])
                    distances, _ = nn.kneighbors(empty_regions)
                    
                    coverage_by_type[node_type] = {
                        'mean_distance_to_gaps': float(np.mean(distances)),
                        'max_distance_to_gaps': float(np.max(distances))
                    }
                
                gaps_info['coverage_by_type'] = coverage_by_type
        
        return gaps_info
    
    def _generate_recommendations(self) -> List[Dict]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Check if synthetic nodes are expanding coverage
        if 'comparisons' in self.analysis_results:
            for comp_name, comp_data in self.analysis_results['comparisons'].items():
                if 'chunk_vs_' in comp_name:
                    node_type = comp_name.replace('chunk_vs_', '')
                    overlap = comp_data['hull_overlap']['overlap_ratio']
                    
                    if overlap > 0.8:
                        recommendations.append({
                            'type': 'high_overlap',
                            'severity': 'medium',
                            'node_type': node_type,
                            'message': f"{node_type} nodes have {overlap:.1%} overlap with chunks. "
                                     f"Consider strategies to explore more diverse regions.",
                            'suggestion': f"Modify {node_type} generation to target low-density areas"
                        })
        
        # Check coverage gaps
        if 'coverage_gaps' in self.analysis_results:
            n_gaps = self.analysis_results['coverage_gaps'].get('n_empty_regions', 0)
            if n_gaps > 10:
                recommendations.append({
                    'type': 'coverage_gaps',
                    'severity': 'high',
                    'message': f"Found {n_gaps} significant empty regions in vector space",
                    'suggestion': "Run targeted REM cycles with diverse node sampling"
                })
        
        # Check diversity within types
        if 'node_metrics' in self.analysis_results:
            for node_type, metrics in self.analysis_results['node_metrics'].items():
                diversity = metrics.get('diversity_index', 0)
                uniformity = metrics.get('uniformity_index', 0)
                
                if diversity < 0.3:  # Low diversity threshold
                    recommendations.append({
                        'type': 'low_diversity',
                        'severity': 'medium',
                        'node_type': node_type,
                        'message': f"{node_type} nodes show low diversity (index: {diversity:.3f})",
                        'suggestion': f"Increase temperature or sampling diversity for {node_type} generation"
                    })
                
                if uniformity < 0.5:  # Poor uniformity
                    recommendations.append({
                        'type': 'poor_uniformity',
                        'severity': 'low',
                        'node_type': node_type,
                        'message': f"{node_type} nodes are unevenly distributed (uniformity: {uniformity:.3f})",
                        'suggestion': "Consider rebalancing sampling strategies"
                    })
        
        return recommendations
    
    def generate_coverage_report(self) -> Dict:
        """Generate a comprehensive coverage report."""
        if not self.analysis_results:
            self.run_full_analysis()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self._generate_summary(),
            'detailed_metrics': self.analysis_results.get('node_metrics', {}),
            'comparisons': self.analysis_results.get('comparisons', {}),
            'coverage_gaps': self.analysis_results.get('coverage_gaps', {}),
            'recommendations': self.analysis_results.get('recommendations', [])
        }
        
        return report
    
    def _generate_summary(self) -> Dict:
        """Generate executive summary of findings."""
        summary = {
            'total_embeddings': sum(
                len(data['embeddings']) 
                for data in self.embeddings_data.values()
            ) if self.embeddings_data else 0,
            'node_types_analyzed': len(self.embeddings_data) if self.embeddings_data else 0
        }
        
        # Key findings
        findings = []
        
        # Check synthetic coverage
        if 'comparisons' in self.analysis_results:
            chunk_vs_synthetic = [
                (k, v) for k, v in self.analysis_results['comparisons'].items() 
                if 'chunk_vs_' in k
            ]
            
            if chunk_vs_synthetic:
                avg_overlap = np.mean([
                    v['hull_overlap']['overlap_ratio'] 
                    for k, v in chunk_vs_synthetic
                ])
                
                if avg_overlap > 0.7:
                    findings.append(
                        f"Synthetic nodes show high overlap ({avg_overlap:.1%}) with original chunks"
                    )
                else:
                    findings.append(
                        f"Synthetic nodes are exploring new regions (overlap: {avg_overlap:.1%})"
                    )
        
        # Coverage gaps
        if 'coverage_gaps' in self.analysis_results:
            n_gaps = self.analysis_results['coverage_gaps'].get('n_empty_regions', 0)
            if n_gaps > 0:
                findings.append(f"Identified {n_gaps} significant coverage gaps")
        
        summary['key_findings'] = findings
        
        return summary
    
    def _save_results(self):
        """Save analysis results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full report as JSON
        report_path = self.output_dir / f"reports/coverage_analysis_{timestamp}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.generate_coverage_report(), f, indent=2)
        
        print(f"\nReport saved to: {report_path}")
        
        # Save coverage gaps as CSV for easy processing
        if 'coverage_gaps' in self.analysis_results:
            gaps = self.analysis_results['coverage_gaps']
            if 'empty_region_centers' in gaps and gaps['empty_region_centers']:
                import pandas as pd
                
                gaps_df = pd.DataFrame(gaps['empty_region_centers'])
                gaps_df.columns = [f'dim_{i}' for i in range(gaps_df.shape[1])]
                
                gaps_path = self.output_dir / f"reports/coverage_gaps_{timestamp}.csv"
                gaps_df.to_csv(gaps_path, index=False)
                print(f"Coverage gaps saved to: {gaps_path}")
    
    def quick_summary(self) -> str:
        """Generate a quick text summary of the analysis."""
        if not self.analysis_results:
            return "No analysis results available. Run analysis first."
        
        summary_lines = [
            "\nQUICK SUMMARY",
            "=" * 40
        ]
        
        # Node counts
        if self.embeddings_data:
            summary_lines.append("\nNode counts:")
            for node_type, data in self.embeddings_data.items():
                count = len(data['embeddings'])
                summary_lines.append(f"  {node_type}: {count}")
        
        # Key metrics
        if 'node_metrics' in self.analysis_results:
            summary_lines.append("\nKey metrics by type:")
            for node_type, metrics in self.analysis_results['node_metrics'].items():
                diversity = metrics.get('diversity_index', 0)
                volume = metrics.get('hull_volume', 0)
                summary_lines.append(
                    f"  {node_type}: diversity={diversity:.3f}, hull_volume={volume:.3f}"
                )
        
        # Recommendations
        if 'recommendations' in self.analysis_results:
            recs = self.analysis_results['recommendations']
            if recs:
                summary_lines.append(f"\nTop recommendations ({len(recs)}):")
                for rec in recs[:3]:
                    summary_lines.append(f"  - {rec['message']}")
        
        return "\n".join(summary_lines)