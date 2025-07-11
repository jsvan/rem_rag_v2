#!/usr/bin/env python3
"""
Generate a comprehensive report from vector space analysis results.

Usage:
    python research/scripts/generate_report.py
    python research/scripts/generate_report.py --format markdown
"""

import sys
import os
from pathlib import Path
import argparse
import json
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_latest_results(reports_dir: Path) -> dict:
    """Load the most recent analysis results."""
    # Find all coverage analysis files
    analysis_files = list(reports_dir.glob("coverage_analysis_*.json"))
    
    if not analysis_files:
        raise FileNotFoundError("No analysis results found. Run analysis first.")
    
    # Get the most recent file
    latest_file = max(analysis_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def generate_markdown_report(results: dict, output_path: Path):
    """Generate a markdown report from analysis results."""
    
    lines = []
    
    # Header
    lines.append("# Vector Space Coverage Analysis Report")
    lines.append(f"\nGenerated: {results.get('timestamp', 'Unknown')}")
    lines.append("\n---\n")
    
    # Executive Summary
    lines.append("## Executive Summary")
    
    summary = results.get('summary', {})
    lines.append(f"\n- **Total Embeddings**: {summary.get('total_embeddings', 0):,}")
    lines.append(f"- **Node Types Analyzed**: {summary.get('node_types_analyzed', 0)}")
    
    if 'key_findings' in summary:
        lines.append("\n### Key Findings:")
        for finding in summary['key_findings']:
            lines.append(f"- {finding}")
    
    # Detailed Metrics
    lines.append("\n## Detailed Metrics by Node Type")
    
    if 'detailed_metrics' in results:
        for node_type, metrics in results['detailed_metrics'].items():
            lines.append(f"\n### {node_type.title()}")
            lines.append(f"- **Count**: {metrics.get('n_samples', 0):,}")
            lines.append(f"- **Diversity Index**: {metrics.get('diversity_index', 0):.4f}")
            lines.append(f"- **Uniformity Index**: {metrics.get('uniformity_index', 0):.4f}")
            lines.append(f"- **Hull Volume**: {metrics.get('hull_volume', 0):.4f}")
            lines.append(f"- **Effective Dimensions**: {metrics.get('effective_dim', 0)}")
            lines.append(f"- **Mean Local Density**: {metrics.get('mean_local_density', 0):.4f}")
            lines.append(f"- **Isolated Points**: {metrics.get('percent_isolated', 0):.1f}%")
    
    # Coverage Analysis
    lines.append("\n## Coverage Analysis")
    
    if 'comparisons' in results:
        lines.append("\n### Hull Overlap Between Node Types")
        
        # Create overlap table
        overlaps = []
        for comp_name, comp_data in results['comparisons'].items():
            if 'hull_overlap' in comp_data:
                type1, type2 = comp_name.split('_vs_')
                overlap = comp_data['hull_overlap']['overlap_ratio']
                overlaps.append({
                    'Type 1': type1,
                    'Type 2': type2,
                    'Overlap': f"{overlap:.1%}"
                })
        
        if overlaps:
            df = pd.DataFrame(overlaps)
            lines.append("\n" + df.to_markdown(index=False))
    
    # Coverage Gaps
    if 'coverage_gaps' in results:
        gaps = results['coverage_gaps']
        lines.append("\n### Coverage Gaps")
        lines.append(f"- **Empty Regions Identified**: {gaps.get('n_empty_regions', 0)}")
        
        if 'coverage_by_type' in gaps:
            lines.append("\n**Distance to Gaps by Node Type:**")
            for node_type, info in gaps['coverage_by_type'].items():
                lines.append(f"- {node_type}: {info['mean_distance_to_gaps']:.4f} (mean)")
    
    # Recommendations
    lines.append("\n## Recommendations")
    
    if 'recommendations' in results:
        # Group by severity
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for rec in results['recommendations']:
            severity = rec.get('severity', 'low')
            if severity == 'high':
                high_priority.append(rec)
            elif severity == 'medium':
                medium_priority.append(rec)
            else:
                low_priority.append(rec)
        
        if high_priority:
            lines.append("\n### High Priority")
            for rec in high_priority:
                lines.append(f"\n**{rec.get('type', 'general')}**: {rec['message']}")
                if 'suggestion' in rec:
                    lines.append(f"\n*Action*: {rec['suggestion']}")
        
        if medium_priority:
            lines.append("\n### Medium Priority")
            for rec in medium_priority:
                lines.append(f"\n**{rec.get('type', 'general')}**: {rec['message']}")
                if 'suggestion' in rec:
                    lines.append(f"\n*Action*: {rec['suggestion']}")
        
        if low_priority:
            lines.append("\n### Low Priority")
            for rec in low_priority:
                lines.append(f"\n- {rec['message']}")
    
    # Technical Details
    lines.append("\n## Technical Details")
    lines.append("\n### Analysis Parameters")
    lines.append("- **Metric**: Cosine distance")
    lines.append("- **PCA Components for Hull**: 10")
    lines.append("- **Local Density k-neighbors**: 10")
    lines.append("- **Coverage Threshold**: 95%")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Markdown report saved to: {output_path}")


def generate_csv_summary(results: dict, output_path: Path):
    """Generate CSV summary of key metrics."""
    
    rows = []
    
    if 'detailed_metrics' in results:
        for node_type, metrics in results['detailed_metrics'].items():
            row = {
                'node_type': node_type,
                'count': metrics.get('n_samples', 0),
                'diversity_index': metrics.get('diversity_index', 0),
                'uniformity_index': metrics.get('uniformity_index', 0),
                'hull_volume': metrics.get('hull_volume', 0),
                'effective_dimensions': metrics.get('effective_dim', 0),
                'mean_local_density': metrics.get('mean_local_density', 0),
                'percent_isolated': metrics.get('percent_isolated', 0),
                'coverage_radius_95': metrics.get('coverage_radius_95', 0)
            }
            rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"CSV summary saved to: {output_path}")


def main():
    """Generate reports from analysis results."""
    
    parser = argparse.ArgumentParser(
        description="Generate reports from vector space analysis"
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "csv", "both"],
        default="both",
        help="Output format for the report"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="research/outputs",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir)
    reports_dir = output_dir / "reports"
    
    # Load latest results
    print("Loading analysis results...")
    try:
        results = load_latest_results(reports_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Generate reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.format in ["markdown", "both"]:
        md_path = reports_dir / f"analysis_report_{timestamp}.md"
        generate_markdown_report(results, md_path)
    
    if args.format in ["csv", "both"]:
        csv_path = reports_dir / f"metrics_summary_{timestamp}.csv"
        generate_csv_summary(results, csv_path)
    
    print("\nReport generation complete!")


if __name__ == "__main__":
    main()