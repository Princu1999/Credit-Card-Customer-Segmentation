from __future__ import annotations
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def correlation_heatmap(df: pd.DataFrame, out_path: str | Path):
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()