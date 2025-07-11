"""
Density visualization functions for vector space analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import Dict, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_density_comparison(embeddings_dict: Dict[str, np.ndarray], 
                          output_path: str,
                          node_types: Optional[List[str]] = None):
    """
    Create density comparison plots for each node type.
    
    Args:
        embeddings_dict: Dict mapping node_type to embeddings array
        output_path: Path to save the plot
        node_types: Optional list of node types to plot
    """
    if node_types is None:
        node_types = list(embeddings_dict.keys())
    
    # Filter to requested types
    plot_data = {k: v for k, v in embeddings_dict.items() if k in node_types}
    
    if not plot_data:
        logger.warning("No data to plot")
        return
    
    # Create figure with subplots
    n_types = len(plot_data)
    fig, axes = plt.subplots(1, n_types, figsize=(5*n_types, 5))
    
    if n_types == 1:
        axes = [axes]
    
    # Project to 2D using PCA
    all_embeddings = np.vstack(list(plot_data.values()))
    pca = PCA(n_components=2)
    pca.fit(all_embeddings)
    
    # Plot each type
    for ax, (node_type, embeddings) in zip(axes, plot_data.items()):
        if len(embeddings) == 0:
            ax.text(0.5, 0.5, f"No {node_type} embeddings", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{node_type} density")
            continue
        
        # Project embeddings
        projected = pca.transform(embeddings)
        
        # Create density plot
        try:
            sns.kdeplot(data=projected, x=0, y=1, ax=ax, fill=True, cmap='viridis')
            ax.scatter(projected[:, 0], projected[:, 1], alpha=0.3, s=10)
        except Exception as e:
            logger.error(f"Error creating density plot for {node_type}: {e}")
            ax.scatter(projected[:, 0], projected[:, 1], alpha=0.5, s=20)
        
        ax.set_title(f"{node_type} density\n(n={len(embeddings)})")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved density comparison to {output_path}")


def plot_distance_distributions(embeddings_dict: Dict[str, np.ndarray], 
                              output_path: str,
                              sample_size: int = 1000):
    """
    Plot pairwise distance distributions for each node type.
    
    Args:
        embeddings_dict: Dict mapping node_type to embeddings array
        output_path: Path to save the plot
        sample_size: Max samples to use for distance calculation
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for node_type, embeddings in embeddings_dict.items():
        if len(embeddings) < 2:
            continue
        
        # Sample if too large
        if len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            embeddings_sample = embeddings[indices]
        else:
            embeddings_sample = embeddings
        
        # Calculate pairwise cosine distances
        from scipy.spatial.distance import pdist
        distances = pdist(embeddings_sample, metric='cosine')
        
        # Plot distribution
        sns.kdeplot(data=distances, label=f"{node_type} (n={len(embeddings)})", 
                   ax=ax, linewidth=2)
    
    ax.set_xlabel("Cosine Distance")
    ax.set_ylabel("Density")
    ax.set_title("Pairwise Distance Distributions by Node Type")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved distance distributions to {output_path}")


def create_density_heatmap(embeddings: np.ndarray, 
                         node_type: str, 
                         output_path: str,
                         bins: int = 50):
    """
    Create 2D density heatmap using PCA projection.
    
    Args:
        embeddings: Embeddings array for a single node type
        node_type: Name of the node type
        output_path: Path to save the plot
        bins: Number of bins for 2D histogram
    """
    if len(embeddings) < 10:
        logger.warning(f"Too few embeddings ({len(embeddings)}) for heatmap")
        return
    
    # Project to 2D
    pca = PCA(n_components=2)
    projected = pca.fit_transform(embeddings)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Hexbin plot
    hb = ax1.hexbin(projected[:, 0], projected[:, 1], gridsize=bins, cmap='YlOrRd')
    ax1.set_title(f"{node_type} Density Hexbin\n(n={len(embeddings)})")
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    cb1 = plt.colorbar(hb, ax=ax1)
    cb1.set_label('Count')
    
    # 2D histogram heatmap
    hist, xedges, yedges = np.histogram2d(projected[:, 0], projected[:, 1], bins=bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    im = ax2.imshow(hist.T, origin='lower', extent=extent, cmap='viridis', aspect='auto')
    ax2.set_title(f"{node_type} Density Heatmap")
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    cb2 = plt.colorbar(im, ax=ax2)
    cb2.set_label('Count')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved density heatmap to {output_path}")


def plot_local_density_comparison(embeddings_dict: Dict[str, np.ndarray],
                                local_densities_dict: Dict[str, np.ndarray],
                                output_path: str):
    """
    Compare local density distributions across node types.
    
    Args:
        embeddings_dict: Dict mapping node_type to embeddings
        local_densities_dict: Dict mapping node_type to local density values
        output_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot of local densities
    density_data = []
    labels = []
    
    for node_type in sorted(local_densities_dict.keys()):
        if node_type in local_densities_dict:
            density_data.append(local_densities_dict[node_type])
            labels.append(f"{node_type}\n(n={len(embeddings_dict[node_type])})")
    
    bp = ax1.boxplot(density_data, labels=labels, patch_artist=True)
    
    # Color boxes
    colors = sns.color_palette("husl", len(density_data))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax1.set_ylabel("Local Density")
    ax1.set_title("Local Density Distribution by Node Type")
    ax1.grid(True, alpha=0.3)
    
    # Violin plot
    positions = range(len(density_data))
    vp = ax2.violinplot(density_data, positions=positions, showmeans=True)
    
    # Color violins
    for pc, color in zip(vp['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Local Density")
    ax2.set_title("Local Density Distributions (Violin Plot)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved local density comparison to {output_path}")


def create_density_summary_plot(analysis_results: Dict, output_path: str):
    """
    Create a summary plot of density metrics.
    
    Args:
        analysis_results: Full analysis results dictionary
        output_path: Path to save the plot
    """
    if 'node_metrics' not in analysis_results:
        logger.warning("No node metrics in analysis results")
        return
    
    metrics = analysis_results['node_metrics']
    node_types = list(metrics.keys())
    
    # Extract metrics
    diversity_indices = []
    uniformity_indices = []
    mean_densities = []
    
    for nt in node_types:
        diversity_indices.append(metrics[nt].get('diversity_index', 0))
        uniformity_indices.append(metrics[nt].get('uniformity_index', 0))
        mean_densities.append(metrics[nt].get('mean_local_density', 0))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Diversity Index
    ax = axes[0, 0]
    bars = ax.bar(node_types, diversity_indices, color=sns.color_palette("husl", len(node_types)))
    ax.set_ylabel("Diversity Index")
    ax.set_title("Spatial Diversity by Node Type")
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, diversity_indices):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom')
    
    # 2. Uniformity Index
    ax = axes[0, 1]
    bars = ax.bar(node_types, uniformity_indices, color=sns.color_palette("husl", len(node_types)))
    ax.set_ylabel("Uniformity Index")
    ax.set_title("Distribution Uniformity by Node Type")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bar, val in zip(bars, uniformity_indices):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom')
    
    # 3. Coverage metrics
    ax = axes[1, 0]
    coverage_data = []
    for nt in node_types:
        coverage_data.append([
            metrics[nt].get('coverage_radius_80', 0),
            metrics[nt].get('coverage_radius_90', 0),
            metrics[nt].get('coverage_radius_95', 0)
        ])
    
    x = np.arange(len(node_types))
    width = 0.25
    
    for i, pct in enumerate(['80%', '90%', '95%']):
        values = [cd[i] for cd in coverage_data]
        ax.bar(x + i*width - width, values, width, label=pct)
    
    ax.set_xlabel("Node Type")
    ax.set_ylabel("Coverage Radius")
    ax.set_title("Coverage Radius by Percentage")
    ax.set_xticks(x)
    ax.set_xticklabels(node_types)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Summary statistics table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary table
    table_data = []
    for nt in node_types:
        table_data.append([
            nt,
            metrics[nt].get('n_samples', 0),
            f"{metrics[nt].get('diversity_index', 0):.3f}",
            f"{metrics[nt].get('uniformity_index', 0):.3f}",
            f"{metrics[nt].get('percent_isolated', 0):.1f}%"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Type', 'Count', 'Diversity', 'Uniformity', 'Isolated'],
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    ax.set_title("Summary Statistics", pad=20)
    
    plt.suptitle("Vector Space Coverage Analysis Summary", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved density summary to {output_path}")