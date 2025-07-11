"""
Coverage visualization functions focusing on convex hulls and spatial coverage.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_convex_hulls_2d(embeddings_dict: Dict[str, np.ndarray], 
                        output_path: str,
                        node_types: Optional[List[str]] = None):
    """
    Plot 2D projections of convex hulls for different node types.
    
    Args:
        embeddings_dict: Dict mapping node_type to embeddings
        output_path: Path to save the plot
        node_types: Optional list of specific node types to plot
    """
    if node_types is None:
        node_types = list(embeddings_dict.keys())
    
    # Filter and combine all embeddings for consistent PCA
    all_embeddings = []
    valid_types = []
    
    for nt in node_types:
        if nt in embeddings_dict and len(embeddings_dict[nt]) >= 3:
            all_embeddings.append(embeddings_dict[nt])
            valid_types.append(nt)
    
    if not all_embeddings:
        logger.warning("Not enough data for convex hull visualization")
        return
    
    # PCA projection
    all_embeddings = np.vstack(all_embeddings)
    pca = PCA(n_components=2)
    pca.fit(all_embeddings)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("husl", len(valid_types))
    
    # Plot each hull
    for i, (node_type, color) in enumerate(zip(valid_types, colors)):
        embeddings = embeddings_dict[node_type]
        projected = pca.transform(embeddings)
        
        # Plot points
        ax.scatter(projected[:, 0], projected[:, 1], 
                  color=color, alpha=0.3, s=20, label=f"{node_type} points")
        
        # Calculate and plot convex hull
        try:
            hull = ConvexHull(projected)
            
            # Plot hull edges
            for simplex in hull.simplices:
                ax.plot(projected[simplex, 0], projected[simplex, 1], 
                       color=color, alpha=0.8, linewidth=2)
            
            # Fill hull
            hull_points = projected[hull.vertices]
            ax.fill(hull_points[:, 0], hull_points[:, 1], 
                   color=color, alpha=0.1, label=f"{node_type} hull")
            
        except Exception as e:
            logger.warning(f"Could not create hull for {node_type}: {e}")
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Convex Hulls by Node Type (2D Projection)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved convex hulls plot to {output_path}")


def plot_hull_metrics_comparison(analysis_results: Dict, output_path: str):
    """
    Compare hull-based metrics across node types.
    
    Args:
        analysis_results: Full analysis results
        output_path: Path to save the plot
    """
    if 'node_metrics' not in analysis_results:
        logger.warning("No node metrics found")
        return
    
    metrics = analysis_results['node_metrics']
    node_types = list(metrics.keys())
    
    # Extract hull metrics
    volumes = []
    surface_areas = []
    compactness = []
    effective_dims = []
    
    for nt in node_types:
        volumes.append(metrics[nt].get('hull_volume', 0))
        surface_areas.append(metrics[nt].get('hull_surface_area', 0))
        compactness.append(metrics[nt].get('compactness', 0))
        effective_dims.append(metrics[nt].get('effective_dim', 0))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = sns.color_palette("husl", len(node_types))
    
    # 1. Hull Volume
    ax = axes[0, 0]
    bars = ax.bar(node_types, volumes, color=colors)
    ax.set_ylabel("Hull Volume")
    ax.set_title("Convex Hull Volume by Node Type")
    ax.grid(True, alpha=0.3)
    
    # 2. Surface Area
    ax = axes[0, 1]
    bars = ax.bar(node_types, surface_areas, color=colors)
    ax.set_ylabel("Surface Area")
    ax.set_title("Convex Hull Surface Area by Node Type")
    ax.grid(True, alpha=0.3)
    
    # 3. Compactness
    ax = axes[1, 0]
    bars = ax.bar(node_types, compactness, color=colors)
    ax.set_ylabel("Compactness (Volume/Surface)")
    ax.set_title("Hull Compactness by Node Type")
    ax.grid(True, alpha=0.3)
    
    # 4. Effective Dimensions
    ax = axes[1, 1]
    bars = ax.bar(node_types, effective_dims, color=colors)
    ax.set_ylabel("Effective Dimensions")
    ax.set_title("Effective Dimensionality by Node Type")
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Convex Hull Analysis", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved hull metrics comparison to {output_path}")


def plot_coverage_overlap_matrix(analysis_results: Dict, output_path: str):
    """
    Create a matrix visualization of hull overlaps between node types.
    
    Args:
        analysis_results: Full analysis results
        output_path: Path to save the plot
    """
    if 'comparisons' not in analysis_results:
        logger.warning("No comparison data found")
        return
    
    comparisons = analysis_results['comparisons']
    
    # Extract node types
    node_types = set()
    for comp_name in comparisons.keys():
        if '_vs_' in comp_name:
            type1, type2 = comp_name.split('_vs_')
            node_types.add(type1)
            node_types.add(type2)
    
    node_types = sorted(list(node_types))
    n_types = len(node_types)
    
    # Create overlap matrix
    overlap_matrix = np.zeros((n_types, n_types))
    np.fill_diagonal(overlap_matrix, 1.0)
    
    for comp_name, comp_data in comparisons.items():
        if '_vs_' in comp_name and 'hull_overlap' in comp_data:
            type1, type2 = comp_name.split('_vs_')
            if type1 in node_types and type2 in node_types:
                i = node_types.index(type1)
                j = node_types.index(type2)
                overlap = comp_data['hull_overlap']['overlap_ratio']
                overlap_matrix[i, j] = overlap
                overlap_matrix[j, i] = overlap
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(overlap_matrix, 
                xticklabels=node_types,
                yticklabels=node_types,
                annot=True, 
                fmt='.2f',
                cmap='YlOrRd',
                vmin=0, vmax=1,
                square=True,
                cbar_kws={'label': 'Overlap Ratio'},
                ax=ax)
    
    ax.set_title("Convex Hull Overlap Matrix")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved overlap matrix to {output_path}")


def plot_coverage_gaps_visualization(embeddings_dict: Dict[str, np.ndarray],
                                   gap_centers: np.ndarray,
                                   output_path: str):
    """
    Visualize coverage gaps in the embedding space.
    
    Args:
        embeddings_dict: Dict mapping node_type to embeddings
        gap_centers: Centers of identified gaps
        output_path: Path to save the plot
    """
    # Combine all embeddings
    all_embeddings = []
    all_labels = []
    
    for node_type, embeddings in embeddings_dict.items():
        all_embeddings.append(embeddings)
        all_labels.extend([node_type] * len(embeddings))
    
    if not all_embeddings:
        logger.warning("No embeddings to visualize")
        return
    
    all_embeddings = np.vstack(all_embeddings)
    
    # Add gap centers if any
    if len(gap_centers) > 0:
        all_embeddings = np.vstack([all_embeddings, gap_centers])
        all_labels.extend(['gap'] * len(gap_centers))
    
    # PCA projection
    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_embeddings)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot embeddings by type
    unique_labels = list(set(all_labels))
    colors = sns.color_palette("husl", len(unique_labels))
    
    for label, color in zip(unique_labels, colors):
        mask = np.array(all_labels) == label
        points = projected[mask]
        
        if label == 'gap':
            # Highlight gaps with different marker
            ax.scatter(points[:, 0], points[:, 1], 
                      color='red', s=200, marker='X', 
                      edgecolors='black', linewidth=2,
                      label=f"Coverage gaps (n={len(points)})",
                      alpha=0.8)
        else:
            ax.scatter(points[:, 0], points[:, 1], 
                      color=color, alpha=0.3, s=20, 
                      label=f"{label} (n={sum(mask)})")
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Coverage Gaps Visualization")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved coverage gaps visualization to {output_path}")


def create_coverage_summary_figure(analysis_results: Dict, output_path: str):
    """
    Create a comprehensive coverage summary figure.
    
    Args:
        analysis_results: Full analysis results
        output_path: Path to save the plot
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Define grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Node type distribution (pie chart)
    ax1 = fig.add_subplot(gs[0, 0])
    if 'node_metrics' in analysis_results:
        counts = [m.get('n_samples', 0) for m in analysis_results['node_metrics'].values()]
        labels = list(analysis_results['node_metrics'].keys())
        
        ax1.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.set_title("Distribution of Node Types")
    
    # 2. Coverage comparison bar chart
    ax2 = fig.add_subplot(gs[0, 1:])
    if 'node_metrics' in analysis_results:
        metrics = analysis_results['node_metrics']
        node_types = list(metrics.keys())
        
        x = np.arange(len(node_types))
        width = 0.35
        
        diversity = [m.get('diversity_index', 0) for m in metrics.values()]
        uniformity = [m.get('uniformity_index', 0) for m in metrics.values()]
        
        bars1 = ax2.bar(x - width/2, diversity, width, label='Diversity')
        bars2 = ax2.bar(x + width/2, uniformity, width, label='Uniformity')
        
        ax2.set_xlabel('Node Type')
        ax2.set_ylabel('Index Value')
        ax2.set_title('Coverage Quality Metrics')
        ax2.set_xticks(x)
        ax2.set_xticklabels(node_types)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Overlap heatmap
    ax3 = fig.add_subplot(gs[1, :2])
    if 'comparisons' in analysis_results:
        # Create simplified overlap data
        overlaps = []
        labels = []
        
        for comp_name, comp_data in analysis_results['comparisons'].items():
            if 'hull_overlap' in comp_data:
                overlaps.append(comp_data['hull_overlap']['overlap_ratio'])
                labels.append(comp_name.replace('_', ' '))
        
        if overlaps:
            y_pos = np.arange(len(labels))
            bars = ax3.barh(y_pos, overlaps)
            
            # Color bars based on overlap level
            for i, (bar, overlap) in enumerate(zip(bars, overlaps)):
                if overlap > 0.8:
                    bar.set_color('red')
                elif overlap > 0.5:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')
            
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(labels)
            ax3.set_xlabel('Overlap Ratio')
            ax3.set_title('Hull Overlap Between Node Types')
            ax3.set_xlim(0, 1)
            ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Key findings text
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    
    findings_text = "Key Findings:\n\n"
    
    if 'recommendations' in analysis_results:
        for i, rec in enumerate(analysis_results['recommendations'][:3]):
            findings_text += f"{i+1}. {rec['message']}\n\n"
    
    ax4.text(0.05, 0.95, findings_text, 
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment='top',
            wrap=True)
    ax4.set_title("Recommendations", fontsize=12)
    
    # 5. Coverage gaps info
    ax5 = fig.add_subplot(gs[2, :])
    if 'coverage_gaps' in analysis_results:
        gaps = analysis_results['coverage_gaps']
        
        gap_text = f"Coverage Analysis:\n"
        gap_text += f"• Empty regions identified: {gaps.get('n_empty_regions', 0)}\n"
        
        if 'coverage_by_type' in gaps:
            gap_text += "\nMean distance to gaps by type:\n"
            for nt, info in gaps['coverage_by_type'].items():
                gap_text += f"  - {nt}: {info['mean_distance_to_gaps']:.3f}\n"
        
        ax5.text(0.05, 0.95, gap_text,
                transform=ax5.transAxes,
                fontsize=11,
                verticalalignment='top',
                fontfamily='monospace')
        ax5.set_title("Coverage Gap Analysis", fontsize=12)
        ax5.axis('off')
    
    plt.suptitle("Vector Space Coverage Analysis Summary", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved coverage summary figure to {output_path}")