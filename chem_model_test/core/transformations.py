"""
Transformation utilities for feature engineering, dimensionality reduction,
and related preprocessing steps (PCA, t-SNE, etc.).
"""

import logging
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import Tuple

logger = logging.getLogger(__name__)


def perform_pca(X: np.ndarray, n_components: int = 2) -> Tuple[PCA, np.ndarray]:
    """
    Perform Principal Component Analysis on the dataset.

    :param X: Feature matrix (num_samples x num_features).
    :param n_components: Number of principal components to keep.
    :return: (PCA object, transformed dataset).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    logger.info(f"PCA performed with {n_components} components, shape of result: {X_pca.shape}")
    return pca, X_pca


def perform_tsne(X: np.ndarray, n_components: int = 2, perplexity: int = 30) -> np.ndarray:
    """
    Perform t-SNE on the dataset.

    :param X: Feature matrix (num_samples x num_features).
    :param n_components: Dimensionality of the embedded space, typically 2.
    :param perplexity: Perplexity parameter for t-SNE (30 is typical).
    :return: The t-SNE-transformed data (num_samples x n_components).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    logger.info(f"t-SNE performed with {n_components} components and perplexity {perplexity}.")
    return X_tsne
