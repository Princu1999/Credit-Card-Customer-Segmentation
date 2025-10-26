from __future__ import annotations
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

from .data import load_dataset, NUMERIC_COLS
from .utils import zscore_outlier_clip, standardize
from .viz import save_elbow_plot, save_pca_scatter_2d, save_heatmap
from .clustering.kmeans import fit_kmeans, kmeans_metrics, inertia_over_k
from .clustering.dbscan import fit_dbscan, dbscan_metrics
from .clustering.gmm import fit_gmm, gmm_metrics
from .clustering.hybrid import majority_vote, centroid_distances, secondary_kmeans

RESULTS = Path("results/figs")
RESULTS.mkdir(parents=True, exist_ok=True)

def run_pipeline(data_path: str | Path = "data/Customer_Data.csv", random_state: int = 42):
    # Load and clean
    df = load_dataset(data_path)
    df = df.dropna(subset=NUMERIC_COLS)  # simple drop NA; replace with smarter imputation if needed
    df = zscore_outlier_clip(df, NUMERIC_COLS, z=3.0)
    X = df[NUMERIC_COLS].copy()

    # Standardize
    X_std, stats = standardize(X, NUMERIC_COLS)

    # KMeans elbow
    ks, inertias = inertia_over_k(X_std, 1, 10, random_state=random_state)
    save_elbow_plot(ks, inertias, RESULTS / "elbow_kmeans.png")    

    # Fit base clusterers
    km, km_labels = fit_kmeans(X_std, k=3, random_state=random_state)
    db, db_labels = fit_dbscan(X_std, eps=3.5, min_samples=40)
    gm, gmm_labels = fit_gmm(X_std, n_components=2, random_state=random_state)

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=random_state).fit(X_std)
    X2 = pd.DataFrame(pca.transform(X_std), columns=["PC1","PC2"])    

    # Save simple 2D plots
    save_pca_scatter_2d(X2, km_labels, RESULTS / "pca_kmeans.png")
    save_pca_scatter_2d(X2, gmm_labels, RESULTS / "pca_gmm.png")

    # Metrics
    m_km = kmeans_metrics(X_std, km_labels)
    m_db = dbscan_metrics(X_std, db_labels)
    m_gm = gmm_metrics(X_std, gmm_labels)

    # Hybrid features + majority vote
    feat = centroid_distances(X_std, km.cluster_centers_, db_labels, gm.means_)
    sec_km, hybrid_labels = secondary_kmeans(feat, k=3, random_state=random_state)

    # Cluster profile heatmap
    df_profile = df.groupby(hybrid_labels).mean(numeric_only=True).sort_index()
    from .viz import save_heatmap
    save_heatmap(df_profile, RESULTS / "cluster_profiles_heatmap.png")

    # Return a summary
    return {
        "kmeans": m_km,
        "dbscan": m_db,
        "gmm": m_gm,
        "hybrid": {"counts": df_profile.shape[0], "features": list(df_profile.columns)},
        "figures": [
            str(RESULTS / "elbow_kmeans.png"),
            str(RESULTS / "pca_kmeans.png"),
            str(RESULTS / "pca_gmm.png"),
            str(RESULTS / "cluster_profiles_heatmap.png"),
        ]
    }

if __name__ == "__main__":
    out = run_pipeline()
    print(out)