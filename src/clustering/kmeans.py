from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def fit_kmeans(X: pd.DataFrame, k: int, random_state: int = 42):
    km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=random_state)
    labels = km.fit_predict(X)
    return km, labels

def kmeans_metrics(X: pd.DataFrame, labels: np.ndarray) -> dict:
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "counts": dict(pd.Series(labels).value_counts())
    }

def inertia_over_k(X: pd.DataFrame, k_min=1, k_max=10, random_state: int = 42):
    inertias, ks = [], []
    for k in range(k_min, k_max+1):
        km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=random_state)
        km.fit(X)
        inertias.append(km.inertia_)
        ks.append(k)
    return ks, inertias