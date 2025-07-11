"""
Dimensionality reduction and projection visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_pca_projections(embeddings_dict: Dict[str, np.ndarray], 
                        output_path: str,
                        n_components: int = 3):
    """
    Create PCA projections with multiple views.
    
    Args:
        embeddings_dict: Dict mapping node_type to embeddings
        output_path: Path to save the plot
        n_components: Number of PCA components to compute
    """
    # Combine all embeddings
    all_embeddings = []
    all_labels = []
    all_types = []
    
    for node_type, embeddings in embeddings_dict.items():
        if len(embeddings) > 0:
            all_embeddings.append(embeddings)
            all_labels.extend([node_type] * len(embeddings))
            all_types.append(node_type)
    
    if not all_embeddings:
        logger.warning("No embeddings to project")
        return
    
    all_embeddings = np.vstack(all_embeddings)
    
    # Perform PCA
    pca = PCA(n_components=min(n_components, all_embeddings.shape[1]))
    projected = pca.fit_transform(all_embeddings)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 2D projections
    pairs = [(0, 1), (0, 2), (1, 2)] if n_components >= 3 else [(0, 1)]
    
    for idx, (i, j) in enumerate(pairs):
        ax = fig.add_subplot(2, 3, idx + 1)
        
        # Plot each type
        for node_type in all_types:
            mask = np.array(all_labels) == node_type
            ax.scatter(projected[mask, i], projected[mask, j], 
                      alpha=0.5, s=20, label=node_type)
        
        ax.set_xlabel(f'PC{i+1} ({pca.explained_variance_ratio_[i]:.1%})')
        ax.set_ylabel(f'PC{j+1} ({pca.explained_variance_ratio_[j]:.1%})')
        ax.set_title(f'PC{i+1} vs PC{j+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Explained variance plot
    ax = fig.add_subplot(2, 3, 4)
    ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
           pca.explained_variance_ratio_)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('PCA Explained Variance')
    ax.grid(True, alpha=0.3)
    
    # Cumulative variance plot
    ax = fig.add_subplot(2, 3, 5)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(range(1, len(cumsum) + 1), cumsum, 'o-')
    ax.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Cumulative Variance Explained')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Component loadings heatmap (top components)
    ax = fig.add_subplot(2, 3, 6)
    n_features_show = min(20, pca.components_.shape[1])
    loadings = pca.components_[:3, :n_features_show]
    
    im = ax.imshow(loadings, aspect='auto', cmap='RdBu_r')
    ax.set_yticks(range(3))
    ax.set_yticklabels([f'PC{i+1}' for i in range(3)])
    ax.set_xlabel('Feature Index')
    ax.set_title('PCA Component Loadings (First 20 features)')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle('PCA Analysis of Embeddings', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved PCA projections to {output_path}")


def plot_tsne_projection(embeddings_dict: Dict[str, np.ndarray], 
                        output_path: str,
                        sample_size: int = 5000,
                        perplexity: int = 30):
    """
    Create t-SNE projection of embeddings.
    
    Args:
        embeddings_dict: Dict mapping node_type to embeddings
        output_path: Path to save the plot
        sample_size: Maximum samples to use (t-SNE is slow)
        perplexity: t-SNE perplexity parameter
    """
    # Combine and sample embeddings
    all_embeddings = []
    all_labels = []
    
    for node_type, embeddings in embeddings_dict.items():
        if len(embeddings) > 0:
            # Sample if too many
            if len(embeddings) > sample_size // len(embeddings_dict):
                n_sample = sample_size // len(embeddings_dict)
                indices = np.random.choice(len(embeddings), n_sample, replace=False)
                embeddings = embeddings[indices]
            
            all_embeddings.append(embeddings)
            all_labels.extend([node_type] * len(embeddings))
    
    if not all_embeddings:
        logger.warning("No embeddings for t-SNE")
        return
    
    all_embeddings = np.vstack(all_embeddings)
    
    # Perform t-SNE
    print(f"Running t-SNE on {len(all_embeddings)} samples...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    projected = tsne.fit_transform(all_embeddings)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each type
    unique_labels = list(set(all_labels))
    colors = sns.color_palette("husl", len(unique_labels))
    
    for label, color in zip(unique_labels, colors):
        mask = np.array(all_labels) == label
        ax.scatter(projected[mask, 0], projected[mask, 1], 
                  color=color, alpha=0.6, s=20, label=label)
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(f't-SNE Projection (perplexity={perplexity}, n={len(all_embeddings)})')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved t-SNE projection to {output_path}")


def plot_umap_projection(embeddings_dict: Dict[str, np.ndarray], 
                        output_path: str,
                        n_neighbors: int = 15,
                        min_dist: float = 0.1):
    """
    Create UMAP projection of embeddings.
    
    Args:
        embeddings_dict: Dict mapping node_type to embeddings
        output_path: Path to save the plot
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
    """
    try:
        # Combine embeddings
        all_embeddings = []
        all_labels = []
        
        for node_type, embeddings in embeddings_dict.items():
            if len(embeddings) > 0:
                all_embeddings.append(embeddings)
                all_labels.extend([node_type] * len(embeddings))
        
        if not all_embeddings:
            logger.warning("No embeddings for UMAP")
            return
        
        all_embeddings = np.vstack(all_embeddings)
        
        # Perform UMAP
        print(f"Running UMAP on {len(all_embeddings)} samples...")
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        projected = reducer.fit_transform(all_embeddings)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot each type
        unique_labels = list(set(all_labels))
        colors = sns.color_palette("husl", len(unique_labels))
        
        for label, color in zip(unique_labels, colors):
            mask = np.array(all_labels) == label
            ax.scatter(projected[mask, 0], projected[mask, 1], 
                      color=color, alpha=0.6, s=20, label=label)
        
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'UMAP Projection (n_neighbors={n_neighbors}, min_dist={min_dist})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved UMAP projection to {output_path}")
        
    except ImportError:
        logger.warning("UMAP not installed. Skipping UMAP projection.")
    except Exception as e:
        logger.error(f"Error creating UMAP projection: {e}")


def plot_temporal_evolution(embeddings_dict: Dict[str, np.ndarray],
                           metadata_dict: Dict[str, List[Dict]],
                           output_path: str):
    """
    Visualize how embeddings evolve over time.
    
    Args:
        embeddings_dict: Dict mapping node_type to embeddings
        metadata_dict: Dict mapping node_type to metadata lists
        output_path: Path to save the plot
    """
    # Extract temporal data
    temporal_data = {}
    
    for node_type in embeddings_dict.keys():
        if node_type not in metadata_dict:
            continue
            
        embeddings = embeddings_dict[node_type]
        metadata = metadata_dict[node_type]
        
        # Extract years
        years = []
        valid_indices = []
        
        for i, meta in enumerate(metadata):
            if 'year' in meta and meta['year'] is not None:
                years.append(meta['year'])
                valid_indices.append(i)
        
        if years:
            temporal_data[node_type] = {
                'embeddings': embeddings[valid_indices],
                'years': np.array(years)
            }
    
    if not temporal_data:
        logger.warning("No temporal data found")
        return
    
    # Combine all for PCA
    all_embeddings = np.vstack([d['embeddings'] for d in temporal_data.values()])
    pca = PCA(n_components=2)
    pca.fit(all_embeddings)
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Plot each node type's temporal evolution
    for idx, (node_type, data) in enumerate(temporal_data.items()):
        if idx >= 4:
            break
            
        ax = axes[idx]
        
        # Project embeddings
        projected = pca.transform(data['embeddings'])
        years = data['years']
        
        # Create scatter plot colored by year
        scatter = ax.scatter(projected[:, 0], projected[:, 1], 
                           c=years, cmap='viridis', alpha=0.6, s=30)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Year')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title(f'{node_type} Evolution Over Time')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(temporal_data), 4):
        axes[idx].set_visible(False)
    
    plt.suptitle('Temporal Evolution of Embeddings', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved temporal evolution plot to {output_path}")


def create_projection_comparison(embeddings_dict: Dict[str, np.ndarray],
                               output_path: str,
                               sample_size: int = 1000):
    """
    Compare different projection methods side by side.
    
    Args:
        embeddings_dict: Dict mapping node_type to embeddings
        output_path: Path to save the plot
        sample_size: Number of samples to use
    """
    # Sample and combine embeddings
    all_embeddings = []
    all_labels = []
    
    for node_type, embeddings in embeddings_dict.items():
        if len(embeddings) > 0:
            n_sample = min(len(embeddings), sample_size // len(embeddings_dict))
            if len(embeddings) > n_sample:
                indices = np.random.choice(len(embeddings), n_sample, replace=False)
                embeddings = embeddings[indices]
            
            all_embeddings.append(embeddings)
            all_labels.extend([node_type] * len(embeddings))
    
    if not all_embeddings:
        return
    
    all_embeddings = np.vstack(all_embeddings)
    
    # Create projections
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Colors for consistency
    unique_labels = list(set(all_labels))
    colors = sns.color_palette("husl", len(unique_labels))
    label_to_color = dict(zip(unique_labels, colors))
    
    # 1. PCA
    ax = axes[0]
    pca = PCA(n_components=2)
    pca_proj = pca.fit_transform(all_embeddings)
    
    for label in unique_labels:
        mask = np.array(all_labels) == label
        ax.scatter(pca_proj[mask, 0], pca_proj[mask, 1],
                  color=label_to_color[label], alpha=0.6, s=20, label=label)
    
    ax.set_title('PCA Projection')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. t-SNE
    ax = axes[1]
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_proj = tsne.fit_transform(all_embeddings)
    
    for label in unique_labels:
        mask = np.array(all_labels) == label
        ax.scatter(tsne_proj[mask, 0], tsne_proj[mask, 1],
                  color=label_to_color[label], alpha=0.6, s=20, label=label)
    
    ax.set_title('t-SNE Projection')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. UMAP (if available)
    ax = axes[2]
    try:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        umap_proj = reducer.fit_transform(all_embeddings)
        
        for label in unique_labels:
            mask = np.array(all_labels) == label
            ax.scatter(umap_proj[mask, 0], umap_proj[mask, 1],
                      color=label_to_color[label], alpha=0.6, s=20, label=label)
        
        ax.set_title('UMAP Projection')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
    except:
        ax.text(0.5, 0.5, 'UMAP not available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('UMAP Projection')
    
    plt.suptitle(f'Projection Method Comparison (n={len(all_embeddings)} samples)', 
                fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved projection comparison to {output_path}")