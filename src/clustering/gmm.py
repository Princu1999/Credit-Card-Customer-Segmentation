from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def fit_gmm(X: pd.DataFrame, n_components: int = 2, random_state: int = 42, covariance_type: str = "full"):
    gm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state)
    labels = gm.fit_predict(X)
    return gm, labels

def gmm_metrics(X: pd.DataFrame, labels: np.ndarray) -> dict:
    metr = {
        "silhouette": float(silhouette_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "counts": dict(pd.Series(labels).value_counts())
    }
    return metr