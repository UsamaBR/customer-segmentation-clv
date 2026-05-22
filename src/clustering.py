"""
Clustering module: preprocess features, run multiple algorithms, evaluate.

Designed to work on the output of build_customer_features() — a per-customer
feature DataFrame indexed by Customer ID.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

# Business-friendly names for K-Means clusters
# IMPORTANT: order matches the cluster_km integer label
KMEANS_CLUSTER_NAMES = {
    0: "Lost / One-Time Lapsed",
    1: "Champions",
    2: "New One-Time Buyers",
    3: "At Risk",
    4: "New & Promising",
}

def prepare_features_for_clustering(
    features: pd.DataFrame,
    impute_value: float | None = None,
) -> pd.DataFrame:
    """Handle NaN values; ready for log+scale pipeline.

    avg_days_between_orders is NaN for single-order customers. We impute
    with a large value (impute_value) so they are clearly separated from
    multi-order customers in feature space.

    Parameters
    ----------
    features : pd.DataFrame
        Per-customer features.
    impute_value : float, optional
        Value to use for NaN imputation. Defaults to max observed + 1 day.

    Returns
    -------
    pd.DataFrame
        Same shape, no NaN values.
    """
    features = features.copy()
    if 'avg_days_between_orders' in features.columns:
        if impute_value is None:
            impute_value = features['avg_days_between_orders'].max() + 1
        features['avg_days_between_orders'] = (
            features['avg_days_between_orders'].fillna(impute_value)
        )
    return features


def build_preprocessing_pipeline() -> Pipeline:
    """sklearn pipeline: log1p transform then standardize.

    Used as the front end of every clustering algorithm so they all see
    the same feature space.
    """
    return Pipeline([
        ('log', FunctionTransformer(np.log1p, validate=True)),
        ('scale', StandardScaler()),
    ])


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: int | None = 5000,
    random_state: int = 42,
) -> dict:
    """Compute internal validation metrics.

    Silhouette, Davies-Bouldin, Calinski-Harabasz. Excludes noise points
    (label = -1) and clusterings with fewer than 2 clusters.

    sample_size limits silhouette computation, which is O(n^2) — slow on
    large data. Sampling preserves the metric's character without the wait.
    """
    mask = labels != -1
    X_valid = X[mask]
    labels_valid = labels[mask]

    n_clusters = len(set(labels_valid))
    if n_clusters < 2 or len(labels_valid) < 10:
        return {
            'n_clusters': n_clusters,
            'silhouette': np.nan,
            'davies_bouldin': np.nan,
            'calinski_harabasz': np.nan,
            'n_noise': int((labels == -1).sum()),
        }

    return {
        'n_clusters': n_clusters,
        'silhouette': silhouette_score(
            X_valid, labels_valid,
            sample_size=sample_size,
            random_state=random_state,
        ),
        'davies_bouldin': davies_bouldin_score(X_valid, labels_valid),
        'calinski_harabasz': calinski_harabasz_score(X_valid, labels_valid),
        'n_noise': int((labels == -1).sum()),
    }


def sweep_kmeans(
    X: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """Run K-Means for each K in k_range. Return metrics per K."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        metrics = evaluate_clustering(X, labels, random_state=random_state)
        metrics['k'] = k
        metrics['inertia'] = km.inertia_
        rows.append(metrics)
    return pd.DataFrame(rows).set_index('k')


def sweep_gmm(
    X: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """Run GMM for each K. Use BIC for selection."""
    rows = []
    for k in k_range:
        gmm = GaussianMixture(
            n_components=k,
            random_state=random_state,
            covariance_type='full',
            n_init=3,
        )
        gmm.fit(X)
        labels = gmm.predict(X)
        metrics = evaluate_clustering(X, labels, random_state=random_state)
        metrics['k'] = k
        metrics['bic'] = gmm.bic(X)
        metrics['aic'] = gmm.aic(X)
        rows.append(metrics)
    return pd.DataFrame(rows).set_index('k')


def run_hdbscan(
    X: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int | None = None,
) -> Tuple[np.ndarray, dict]:
    """Fit HDBSCAN with given parameters; return labels and metrics."""
    if not HAS_HDBSCAN:
        # Fallback to DBSCAN if hdbscan isn't installed
        labels = DBSCAN(eps=0.5, min_samples=min_cluster_size).fit_predict(X)
    else:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
        labels = clusterer.fit_predict(X)
    metrics = evaluate_clustering(X, labels)
    return labels, metrics