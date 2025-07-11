"""
Vector space metrics for analyzing coverage and clustering.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VectorSpaceMetrics:
    """Calculate various metrics for analyzing vector space coverage."""
    
    def __init__(self, metric: str = 'cosine'):
        """
        Initialize metrics calculator.
        
        Args:
            metric: Distance metric to use ('cosine', 'euclidean')
        """
        self.metric = metric
    
    def calculate_local_density(self, embeddings: np.ndarray, k: int = 10) -> np.ndarray:
        """
        Calculate local density for each point using k-nearest neighbors.
        
        Local density is defined as the inverse of the average distance
        to the k nearest neighbors.
        
        Args:
            embeddings: Array of embeddings (n_samples, n_features)
            k: Number of nearest neighbors
            
        Returns:
            Array of local density values for each point
        """
        if len(embeddings) <= k:
            logger.warning(f"Not enough samples ({len(embeddings)}) for k={k} neighbors")
            k = max(1, len(embeddings) - 1)
        
        # Fit nearest neighbors
        nn = NearestNeighbors(n_neighbors=k+1, metric=self.metric)
        nn.fit(embeddings)
        
        # Get distances to k nearest neighbors (excluding self)
        distances, _ = nn.kneighbors(embeddings)
        distances = distances[:, 1:]  # Remove self-distance
        
        # Calculate average distance to k neighbors
        avg_distances = np.mean(distances, axis=1)
        
        # Local density is inverse of average distance
        # Add small epsilon to avoid division by zero
        local_densities = 1.0 / (avg_distances + 1e-10)
        
        return local_densities
    
    def calculate_coverage_radius(self, embeddings: np.ndarray, 
                                coverage_percent: float = 0.95) -> float:
        """
        Find radius that covers X% of the space.
        
        This is calculated as the distance within which coverage_percent
        of all pairwise distances fall.
        
        Args:
            embeddings: Array of embeddings
            coverage_percent: Percentage of space to cover (0-1)
            
        Returns:
            Coverage radius
        """
        if len(embeddings) < 2:
            return 0.0
        
        # Calculate all pairwise distances
        distances = pdist(embeddings, metric=self.metric)
        
        # Find the distance percentile
        radius = np.percentile(distances, coverage_percent * 100)
        
        return radius
    
    def calculate_diversity_index(self, embeddings: np.ndarray) -> float:
        """
        Calculate spatial diversity using average pairwise distance.
        
        Higher values indicate more diverse/spread out points.
        
        Args:
            embeddings: Array of embeddings
            
        Returns:
            Diversity index
        """
        if len(embeddings) < 2:
            return 0.0
        
        # Calculate all pairwise distances
        distances = pdist(embeddings, metric=self.metric)
        
        # Return average pairwise distance
        return np.mean(distances)
    
    def calculate_entropy(self, embeddings: np.ndarray, n_bins: int = 50) -> float:
        """
        Calculate spatial entropy of distribution.
        
        Higher entropy indicates more uniform coverage.
        
        Args:
            embeddings: Array of embeddings
            n_bins: Number of bins for discretization
            
        Returns:
            Spatial entropy value
        """
        if len(embeddings) < 2:
            return 0.0
        
        # Project to 2D for binning (using first 2 principal components)
        from sklearn.decomposition import PCA
        
        if embeddings.shape[1] > 2:
            pca = PCA(n_components=2)
            projected = pca.fit_transform(embeddings)
        else:
            projected = embeddings
        
        # Create 2D histogram
        hist, _, _ = np.histogram2d(
            projected[:, 0], 
            projected[:, 1], 
            bins=n_bins
        )
        
        # Flatten and normalize
        hist_flat = hist.flatten()
        hist_norm = hist_flat / hist_flat.sum()
        
        # Calculate entropy
        # Remove zero entries to avoid log(0)
        hist_norm = hist_norm[hist_norm > 0]
        
        return entropy(hist_norm)
    
    def calculate_uniformity_index(self, embeddings: np.ndarray, k: int = 10) -> float:
        """
        Calculate uniformity index based on variance of local densities.
        
        Lower variance in local densities indicates more uniform distribution.
        Returns a value between 0 and 1, where 1 is perfectly uniform.
        
        Args:
            embeddings: Array of embeddings
            k: Number of nearest neighbors for density calculation
            
        Returns:
            Uniformity index (0-1)
        """
        if len(embeddings) <= k:
            return 0.0
        
        # Calculate local densities
        densities = self.calculate_local_density(embeddings, k)
        
        # Calculate coefficient of variation (CV)
        # CV = std / mean
        cv = np.std(densities) / (np.mean(densities) + 1e-10)
        
        # Convert to uniformity index (1 - normalized CV)
        # Normalize CV to [0, 1] range using sigmoid-like function
        uniformity = 1.0 / (1.0 + cv)
        
        return uniformity
    
    def calculate_isolation_score(self, embeddings: np.ndarray, 
                                 threshold_percentile: float = 90) -> Dict:
        """
        Calculate isolation scores to find outliers/isolated points.
        
        Args:
            embeddings: Array of embeddings
            threshold_percentile: Percentile for determining isolation threshold
            
        Returns:
            Dict with isolation statistics and isolated point indices
        """
        if len(embeddings) < 2:
            return {
                'mean_isolation': 0.0,
                'isolated_indices': [],
                'isolation_scores': np.array([])
            }
        
        # Calculate nearest neighbor distances
        nn = NearestNeighbors(n_neighbors=2, metric=self.metric)
        nn.fit(embeddings)
        distances, _ = nn.kneighbors(embeddings)
        
        # Isolation score is distance to nearest neighbor
        isolation_scores = distances[:, 1]
        
        # Find threshold for isolation
        threshold = np.percentile(isolation_scores, threshold_percentile)
        
        # Find isolated points
        isolated_indices = np.where(isolation_scores > threshold)[0]
        
        return {
            'mean_isolation': np.mean(isolation_scores),
            'max_isolation': np.max(isolation_scores),
            'isolation_threshold': threshold,
            'isolated_indices': isolated_indices.tolist(),
            'isolation_scores': isolation_scores,
            'percent_isolated': len(isolated_indices) / len(embeddings) * 100
        }
    
    def calculate_all_metrics(self, embeddings: np.ndarray) -> Dict:
        """
        Calculate all metrics for a set of embeddings.
        
        Args:
            embeddings: Array of embeddings
            
        Returns:
            Dict with all calculated metrics
        """
        if len(embeddings) == 0:
            return {
                'n_samples': 0,
                'error': 'No embeddings provided'
            }
        
        metrics = {
            'n_samples': len(embeddings),
            'embedding_dim': embeddings.shape[1]
        }
        
        if len(embeddings) >= 2:
            # Basic metrics
            metrics['diversity_index'] = self.calculate_diversity_index(embeddings)
            metrics['coverage_radius_95'] = self.calculate_coverage_radius(embeddings, 0.95)
            metrics['coverage_radius_90'] = self.calculate_coverage_radius(embeddings, 0.90)
            metrics['coverage_radius_80'] = self.calculate_coverage_radius(embeddings, 0.80)
            
            # Density metrics
            local_densities = self.calculate_local_density(embeddings)
            metrics['mean_local_density'] = np.mean(local_densities)
            metrics['std_local_density'] = np.std(local_densities)
            metrics['min_local_density'] = np.min(local_densities)
            metrics['max_local_density'] = np.max(local_densities)
            
            # Distribution metrics
            metrics['spatial_entropy'] = self.calculate_entropy(embeddings)
            metrics['uniformity_index'] = self.calculate_uniformity_index(embeddings)
            
            # Isolation metrics
            isolation_info = self.calculate_isolation_score(embeddings)
            metrics['mean_isolation_score'] = isolation_info['mean_isolation']
            metrics['percent_isolated'] = isolation_info['percent_isolated']
        
        return metrics
    
    def compare_distributions(self, embeddings1: np.ndarray, 
                            embeddings2: np.ndarray) -> Dict:
        """
        Compare two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Dict with comparison metrics
        """
        metrics1 = self.calculate_all_metrics(embeddings1)
        metrics2 = self.calculate_all_metrics(embeddings2)
        
        comparison = {
            'set1_metrics': metrics1,
            'set2_metrics': metrics2
        }
        
        # Calculate overlap metrics if both sets have samples
        if len(embeddings1) > 0 and len(embeddings2) > 0:
            # Find nearest neighbor from set1 to set2
            nn = NearestNeighbors(n_neighbors=1, metric=self.metric)
            nn.fit(embeddings2)
            distances, _ = nn.kneighbors(embeddings1)
            
            comparison['mean_nn_distance_1to2'] = np.mean(distances)
            comparison['median_nn_distance_1to2'] = np.median(distances)
            
            # Reverse direction
            nn.fit(embeddings1)
            distances, _ = nn.kneighbors(embeddings2)
            
            comparison['mean_nn_distance_2to1'] = np.mean(distances)
            comparison['median_nn_distance_2to1'] = np.median(distances)
        
        return comparison