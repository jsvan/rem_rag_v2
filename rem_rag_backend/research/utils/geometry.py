"""
Geometric utilities for convex hull and spatial analysis.
"""

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ConvexHullAnalysis:
    """Analyze convex hulls in high-dimensional spaces."""
    
    def __init__(self, n_components: int = 10):
        """
        Initialize convex hull analyzer.
        
        Args:
            n_components: Number of PCA components for dimensionality reduction
        """
        self.n_components = n_components
        
    def _reduce_dimensions(self, points: np.ndarray) -> Tuple[np.ndarray, PCA]:
        """
        Reduce dimensionality for convex hull calculation.
        
        ConvexHull works best in lower dimensions, so we use PCA
        to reduce while preserving most variance.
        
        Args:
            points: High-dimensional points
            
        Returns:
            Tuple of (reduced points, fitted PCA)
        """
        n_samples, n_features = points.shape
        
        # Use min of n_components, n_features, n_samples-1
        n_comp = min(self.n_components, n_features, n_samples - 1)
        
        if n_comp < 3:
            logger.warning(f"Too few components ({n_comp}) for meaningful hull analysis")
            return points[:, :n_comp], None
        
        pca = PCA(n_components=n_comp)
        reduced = pca.fit_transform(points)
        
        variance_explained = np.sum(pca.explained_variance_ratio_)
        logger.info(f"PCA with {n_comp} components explains {variance_explained:.2%} of variance")
        
        return reduced, pca
    
    def calculate_hull_volume(self, points: np.ndarray) -> float:
        """
        Calculate approximate convex hull volume in high dimensions.
        
        For high-dimensional spaces, we use PCA to reduce dimensions
        while preserving most of the variance, then calculate volume
        in the reduced space.
        
        Args:
            points: Array of points (n_samples, n_features)
            
        Returns:
            Approximate hull volume (scaled by explained variance)
        """
        if len(points) < 4:
            logger.warning("Need at least 4 points for hull volume")
            return 0.0
        
        # Reduce dimensions
        reduced_points, pca = self._reduce_dimensions(points)
        
        if reduced_points.shape[1] < 3:
            logger.warning("Not enough dimensions for volume calculation")
            return 0.0
        
        try:
            # Calculate convex hull in reduced space
            hull = ConvexHull(reduced_points)
            
            # Scale volume by explained variance to account for lost dimensions
            if pca is not None:
                variance_factor = np.sum(pca.explained_variance_ratio_)
                scaled_volume = hull.volume * variance_factor
            else:
                scaled_volume = hull.volume
            
            return scaled_volume
            
        except Exception as e:
            logger.error(f"Error calculating hull volume: {e}")
            return 0.0
    
    def calculate_hull_surface_area(self, points: np.ndarray) -> float:
        """
        Calculate approximate convex hull surface area.
        
        Args:
            points: Array of points
            
        Returns:
            Approximate hull surface area
        """
        if len(points) < 4:
            return 0.0
        
        # Reduce dimensions
        reduced_points, pca = self._reduce_dimensions(points)
        
        if reduced_points.shape[1] < 3:
            return 0.0
        
        try:
            hull = ConvexHull(reduced_points)
            
            # Scale by explained variance
            if pca is not None:
                variance_factor = np.sum(pca.explained_variance_ratio_)
                scaled_area = hull.area * variance_factor
            else:
                scaled_area = hull.area
            
            return scaled_area
            
        except Exception as e:
            logger.error(f"Error calculating hull surface area: {e}")
            return 0.0
    
    def calculate_hull_overlap(self, points1: np.ndarray, 
                             points2: np.ndarray) -> Dict[str, float]:
        """
        Calculate overlap between two convex hulls.
        
        This uses a Monte Carlo approach: sample points and check
        which hulls they fall into.
        
        Args:
            points1: First set of points
            points2: Second set of points
            
        Returns:
            Dict with overlap metrics
        """
        if len(points1) < 4 or len(points2) < 4:
            return {
                'overlap_ratio': 0.0,
                'points1_in_hull2': 0.0,
                'points2_in_hull1': 0.0
            }
        
        # Combine points for consistent dimensionality reduction
        all_points = np.vstack([points1, points2])
        
        # Reduce dimensions
        reduced_all, pca = self._reduce_dimensions(all_points)
        
        if reduced_all.shape[1] < 3:
            return {
                'overlap_ratio': 0.0,
                'points1_in_hull2': 0.0,
                'points2_in_hull1': 0.0
            }
        
        # Split back into two sets
        n1 = len(points1)
        reduced_points1 = reduced_all[:n1]
        reduced_points2 = reduced_all[n1:]
        
        try:
            # Create Delaunay triangulations for point-in-hull tests
            hull1 = Delaunay(reduced_points1)
            hull2 = Delaunay(reduced_points2)
            
            # Check how many points from each set are in the other hull
            points1_in_hull2 = np.sum(hull2.find_simplex(reduced_points1) >= 0)
            points2_in_hull1 = np.sum(hull1.find_simplex(reduced_points2) >= 0)
            
            # Calculate ratios
            ratio1_in_2 = points1_in_hull2 / len(points1)
            ratio2_in_1 = points2_in_hull1 / len(points2)
            
            # Overall overlap estimate
            overlap_ratio = (ratio1_in_2 + ratio2_in_1) / 2
            
            return {
                'overlap_ratio': overlap_ratio,
                'points1_in_hull2': ratio1_in_2,
                'points2_in_hull1': ratio2_in_1
            }
            
        except Exception as e:
            logger.error(f"Error calculating hull overlap: {e}")
            return {
                'overlap_ratio': 0.0,
                'points1_in_hull2': 0.0,
                'points2_in_hull1': 0.0
            }
    
    def find_empty_regions(self, all_points: np.ndarray, 
                          resolution: int = 100,
                          coverage_threshold: float = 0.1) -> List[np.ndarray]:
        """
        Identify low-density regions in the space.
        
        Uses a grid-based approach in reduced dimensions.
        
        Args:
            all_points: All points in the space
            resolution: Grid resolution per dimension
            coverage_threshold: Min density to consider region "covered"
            
        Returns:
            List of center points for empty regions
        """
        if len(all_points) < 10:
            return []
        
        # Reduce to 3D for visualization and analysis
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(all_points)
        
        # Create bounding box
        mins = np.min(reduced, axis=0)
        maxs = np.max(reduced, axis=0)
        ranges = maxs - mins
        
        # Add padding
        mins -= 0.1 * ranges
        maxs += 0.1 * ranges
        
        # Create grid
        grid_points = []
        steps = [np.linspace(mins[i], maxs[i], resolution) for i in range(3)]
        
        for x in steps[0][::10]:  # Sample every 10th point for efficiency
            for y in steps[1][::10]:
                for z in steps[2][::10]:
                    grid_points.append([x, y, z])
        
        grid_points = np.array(grid_points)
        
        # Calculate density at each grid point
        from sklearn.neighbors import KernelDensity
        
        kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
        kde.fit(reduced)
        
        log_densities = kde.score_samples(grid_points)
        densities = np.exp(log_densities)
        
        # Normalize densities
        densities = densities / np.max(densities)
        
        # Find low-density regions
        empty_mask = densities < coverage_threshold
        empty_grid_points = grid_points[empty_mask]
        
        # Transform back to original space
        if len(empty_grid_points) > 0:
            # Sample up to 100 empty regions
            if len(empty_grid_points) > 100:
                indices = np.random.choice(len(empty_grid_points), 100, replace=False)
                empty_grid_points = empty_grid_points[indices]
            
            # Transform back
            empty_regions = pca.inverse_transform(empty_grid_points)
            return empty_regions
        
        return []
    
    def calculate_effective_dimension(self, points: np.ndarray, 
                                    variance_threshold: float = 0.95) -> int:
        """
        Calculate the effective dimensionality of the point cloud.
        
        This is the number of dimensions needed to explain
        variance_threshold of the total variance.
        
        Args:
            points: Array of points
            variance_threshold: Cumulative variance threshold
            
        Returns:
            Effective number of dimensions
        """
        if len(points) < 2:
            return 0
        
        # Perform PCA
        n_components = min(points.shape[0] - 1, points.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(points)
        
        # Find number of components for threshold
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_effective = np.argmax(cumsum >= variance_threshold) + 1
        
        return n_effective
    
    def analyze_hull_properties(self, points: np.ndarray) -> Dict:
        """
        Comprehensive analysis of convex hull properties.
        
        Args:
            points: Array of points
            
        Returns:
            Dict with various hull metrics
        """
        if len(points) < 4:
            return {
                'error': 'Not enough points for hull analysis',
                'n_points': len(points)
            }
        
        results = {
            'n_points': len(points),
            'original_dim': points.shape[1]
        }
        
        # Effective dimensionality
        results['effective_dim'] = self.calculate_effective_dimension(points)
        
        # Hull volume and area
        results['hull_volume'] = self.calculate_hull_volume(points)
        results['hull_surface_area'] = self.calculate_hull_surface_area(points)
        
        # Compactness ratio (volume/surface area)
        if results['hull_surface_area'] > 0:
            results['compactness'] = results['hull_volume'] / results['hull_surface_area']
        else:
            results['compactness'] = 0.0
        
        # Point distribution within hull
        reduced_points, pca = self._reduce_dimensions(points)
        
        if reduced_points.shape[1] >= 3:
            try:
                hull = ConvexHull(reduced_points)
                
                # Fraction of points on hull
                n_hull_vertices = len(np.unique(hull.vertices))
                results['fraction_on_hull'] = n_hull_vertices / len(points)
                
                # Hull efficiency (actual volume vs bounding box)
                bbox_volume = np.prod(np.max(reduced_points, axis=0) - 
                                    np.min(reduced_points, axis=0))
                results['hull_efficiency'] = hull.volume / bbox_volume if bbox_volume > 0 else 0
                
            except Exception as e:
                logger.error(f"Error in hull analysis: {e}")
                results['fraction_on_hull'] = 0.0
                results['hull_efficiency'] = 0.0
        
        return results