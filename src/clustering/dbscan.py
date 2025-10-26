from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def fit_dbscan(X: pd.DataFrame, eps: float = 3.5, min_samples: int = 40):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return model, labels

def dbscan_metrics(X: pd.DataFrame, labels: np.ndarray) -> dict:
    # silhouette requires >1 cluster and all finite labels (ignore noise=-1 by masking if necessary)
    mask = labels != -1
    metr = {"counts": dict(pd.Series(labels).value_counts())}
    if mask.sum() > 1 and len(np.unique(labels[mask])) > 1:
        metr["silhouette"] = float(silhouette_score(X[mask], labels[mask]))
        metr["calinski_harabasz"] = float(calinski_harabasz_score(X[mask], labels[mask]))
    return metr