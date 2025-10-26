from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def save_elbow_plot(k_values, inertias, out_path: str | Path):
    plt.figure()
    plt.plot(k_values, inertias, marker='o')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Plot (K-Means)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def save_pca_scatter_2d(df_2d: pd.DataFrame, labels, out_path: str | Path, centroids_2d=None):
    plt.figure()
    sc = plt.scatter(df_2d.iloc[:,0], df_2d.iloc[:,1], c=labels, s=10, alpha=0.8)
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('PCA 2D — Clusters')
    if centroids_2d is not None:
        plt.scatter(centroids_2d[:,0], centroids_2d[:,1], marker='x', s=120)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def save_heatmap(means: pd.DataFrame, out_path: str | Path):
    plt.figure(figsize=(10,6))
    sns.heatmap(means, annot=False, cmap='viridis')
    plt.title('Cluster Profiles Heatmap')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()