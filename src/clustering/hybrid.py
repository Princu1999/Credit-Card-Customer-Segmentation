from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def majority_vote(km_labels: np.ndarray, db_labels: np.ndarray, gmm_labels: np.ndarray) -> np.ndarray:
    # Map noise (-1) to a distinct label to avoid dominating majority
    votes = np.vstack([km_labels, np.where(db_labels==-1, 9999, db_labels), gmm_labels]).T
    out = []
    for row in votes:
        vals, counts = np.unique(row, return_counts=True)
        # ignore the special noise label if possible
        if 9999 in vals and len(vals) > 1:
            mask = vals != 9999
            vals, counts = vals[mask], counts[mask]
        out.append(vals[np.argmax(counts)])
    return np.array(out)

def centroid_distances(X: pd.DataFrame, km_centers, db_labels, gmm_means):
    # distances to KMeans and GMM centroids; for DBSCAN, distance to cluster mean; noise gets large value
    from scipy.spatial.distance import cdist
    d_km = cdist(X, km_centers)
    # DBSCAN: compute cluster means for non-noise, then assign distance to its cluster mean else large value
    uniq = np.unique(db_labels[db_labels!=-1])
    cluster_means = {lab: X[db_labels==lab].mean(axis=0).values for lab in uniq}
    d_db = np.zeros((len(X), len(uniq)))
    for i, lab in enumerate(uniq):
        d_db[:, i] = np.linalg.norm(X.values - cluster_means[lab], axis=1)
    # For simplicity, take min distance across DBSCAN clusters; noise gets a large constant
    min_d_db = d_db.min(axis=1) if d_db.size else np.full(len(X), 1e6)
    min_d_db = np.where(db_labels==-1, 1e6, min_d_db)
    d_gmm = cdist(X, gmm_means)
    # Combine selected distances as features
    return pd.DataFrame({
        "dist_km_min": d_km.min(axis=1),
        "dist_gmm_min": d_gmm.min(axis=1),
        "dist_db_min": min_d_db
    })

def secondary_kmeans(features: pd.DataFrame, k: int = 3, random_state: int = 42):
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(features)
    return km, labels