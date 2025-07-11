"""
Statistical utilities for vector space analysis.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def calculate_distribution_stats(values: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for a distribution.
    
    Args:
        values: Array of values
        
    Returns:
        Dict with various statistics
    """
    if len(values) == 0:
        return {}
    
    return {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75)),
        'iqr': float(np.percentile(values, 75) - np.percentile(values, 25)),
        'skewness': float(stats.skew(values)),
        'kurtosis': float(stats.kurtosis(values))
    }


def compare_distributions(dist1: np.ndarray, dist2: np.ndarray) -> Dict[str, float]:
    """
    Compare two distributions using various statistical tests.
    
    Args:
        dist1: First distribution
        dist2: Second distribution
        
    Returns:
        Dict with test results
    """
    results = {}
    
    if len(dist1) < 2 or len(dist2) < 2:
        return results
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = stats.ks_2samp(dist1, dist2)
    results['ks_statistic'] = float(ks_stat)
    results['ks_pvalue'] = float(ks_pvalue)
    
    # Mann-Whitney U test
    mw_stat, mw_pvalue = stats.mannwhitneyu(dist1, dist2, alternative='two-sided')
    results['mannwhitney_statistic'] = float(mw_stat)
    results['mannwhitney_pvalue'] = float(mw_pvalue)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(dist1) - 1) * np.std(dist1)**2 + 
                          (len(dist2) - 1) * np.std(dist2)**2) / 
                         (len(dist1) + len(dist2) - 2))
    
    if pooled_std > 0:
        cohens_d = (np.mean(dist1) - np.mean(dist2)) / pooled_std
        results['cohens_d'] = float(cohens_d)
    
    return results


def calculate_overlap_coefficient(dist1: np.ndarray, dist2: np.ndarray, 
                                bins: int = 50) -> float:
    """
    Calculate the overlap coefficient between two distributions.
    
    The overlap coefficient is the area of intersection between
    two probability distributions.
    
    Args:
        dist1: First distribution
        dist2: Second distribution
        bins: Number of bins for histogram
        
    Returns:
        Overlap coefficient (0-1)
    """
    if len(dist1) == 0 or len(dist2) == 0:
        return 0.0
    
    # Create histograms with same bins
    min_val = min(np.min(dist1), np.min(dist2))
    max_val = max(np.max(dist1), np.max(dist2))
    
    hist1, bin_edges = np.histogram(dist1, bins=bins, range=(min_val, max_val), 
                                   density=True)
    hist2, _ = np.histogram(dist2, bins=bins, range=(min_val, max_val), 
                           density=True)
    
    # Normalize to probabilities
    bin_width = bin_edges[1] - bin_edges[0]
    hist1 = hist1 * bin_width
    hist2 = hist2 * bin_width
    
    # Calculate overlap
    overlap = np.minimum(hist1, hist2).sum()
    
    return float(overlap)


def calculate_jensen_shannon_divergence(dist1: np.ndarray, dist2: np.ndarray,
                                      bins: int = 50) -> float:
    """
    Calculate Jensen-Shannon divergence between two distributions.
    
    JS divergence is a symmetric measure of the difference between
    two probability distributions.
    
    Args:
        dist1: First distribution
        dist2: Second distribution
        bins: Number of bins for histogram
        
    Returns:
        JS divergence (0 = identical, 1 = maximally different)
    """
    if len(dist1) == 0 or len(dist2) == 0:
        return 1.0
    
    # Create histograms
    min_val = min(np.min(dist1), np.min(dist2))
    max_val = max(np.max(dist1), np.max(dist2))
    
    hist1, _ = np.histogram(dist1, bins=bins, range=(min_val, max_val))
    hist2, _ = np.histogram(dist2, bins=bins, range=(min_val, max_val))
    
    # Normalize
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    hist1 = hist1 + epsilon
    hist2 = hist2 + epsilon
    
    # Calculate JS divergence
    m = 0.5 * (hist1 + hist2)
    js_div = 0.5 * stats.entropy(hist1, m) + 0.5 * stats.entropy(hist2, m)
    
    # Normalize to [0, 1]
    js_div = js_div / np.log(2)
    
    return float(js_div)


def bootstrap_confidence_interval(data: np.ndarray, 
                                statistic_func: callable,
                                confidence: float = 0.95,
                                n_bootstrap: int = 1000) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Args:
        data: Input data
        statistic_func: Function to calculate statistic
        confidence: Confidence level (e.g., 0.95 for 95%)
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(data) == 0:
        return (0.0, 0.0)
    
    bootstrap_stats = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        stat = statistic_func(sample)
        bootstrap_stats.append(stat)
    
    # Calculate percentiles
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha/2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha/2))
    
    return (float(lower), float(upper))