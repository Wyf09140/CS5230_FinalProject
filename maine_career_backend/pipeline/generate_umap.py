"""
Maine Career Wealth Navigator - UMAP 2D Coordinates
Middle Layer: Visualization data pipeline

Generates 2D coordinates for the Wealth Galaxy map using UMAP dimensionality
reduction on the blended feature matrix. Output is a JSON file consumed by
the React/Plotly frontend to render the interactive scatter plot.

Run after embed_and_cluster.py:
    python pipeline/generate_umap.py

Output:
    models/umap_coords.json   - x/y coordinates + metadata per occupation
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


def generate_umap_coords(
    models_dir: str = "models",
    data_dir: str = "data",
    n_neighbors: int = 5,
    min_dist: float = 0.3,
    random_state: int = 42,
):
    """
    Load feature matrix -> UMAP 2D -> save JSON for frontend.
    Falls back to PCA if umap-learn is not installed.
    """
    models_path = Path(models_dir)

    # Load artifacts
    feature_matrix = np.load(models_path / "feature_matrix.npy")

    with open(models_path / "occupations_embedded.pkl", "rb") as f:
        df = pickle.load(f)

    with open(models_path / "pipeline_meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    with open(models_path / "clustering_meta.json", encoding="utf-8") as f:
        clustering_meta = json.load(f)

    print(f"[UMAP] Feature matrix: {feature_matrix.shape}")

    # Choose best cluster column dynamically
    best_method = clustering_meta.get("best_method_hints", {}).get("overall", "kmeans")

    cluster_col_map = {
        "kmeans": "cluster_kmeans",
        "gmm": "cluster_gmm",
        "agglomerative": "cluster_agglom",
        "hdbscan": "cluster_hdbscan",
    }

    cluster_col = cluster_col_map.get(best_method, "cluster_kmeans")

    if cluster_col not in df.columns:
        raise ValueError(
            f"Expected cluster column '{cluster_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Standardize to one output field for frontend
    df = df.copy()
    df["cluster_id"] = df[cluster_col]

    # -- Dimensionality reduction -----------------------------------------------
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            metric="cosine",
        )
        coords_2d = reducer.fit_transform(feature_matrix)
        method = "UMAP"
        print(f"[UMAP] Used UMAP. Shape: {coords_2d.shape}")
    except ImportError:
        print("[UMAP] umap-learn not installed - falling back to PCA (install umap-learn for production)")
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=random_state)
        coords_2d = pca.fit_transform(feature_matrix)
        method = "PCA"
        print(f"[UMAP] Used PCA. Shape: {coords_2d.shape}")

    # -- Build output JSON ------------------------------------------------------
    tier_colors = {
        "Tier 1 - Entry/Support": "#94a3b8",
        "Tier 2 - Technical Core": "#60a5fa",
        "Tier 3 - Senior/Specialist": "#34d399",
        "Tier 4 - Leadership/Expert": "#fbbf24",
    }

    points = []
    for i, row in df.iterrows():
        points.append({
            "soc_code": row["soc_code"],
            "occ_title": row["occ_title"],
            "role_description": row["role_description"],
            "median_wage_me": row["median_wage_me"],
            "wealth_tier": row["wealth_tier"],
            "cluster_id": int(row["cluster_id"]),
            "cluster_method": best_method,
            "x": round(float(coords_2d[i, 0]), 4),
            "y": round(float(coords_2d[i, 1]), 4),
            "color": tier_colors.get(row["wealth_tier"], "#e2e8f0"),
            "size": _wage_to_bubble_size(row["median_wage_me"]),
        })

    output = {
        "method": method,
        "cluster_method": best_method,
        "cluster_column": cluster_col,
        "n_occupations": len(points),
        "k_clusters": meta.get("k_clusters"),
        "tier_colors": tier_colors,
        "tier_wage_thresholds": meta.get("tier_wage_thresholds"),
        "points": points,
    }

    out_path = models_path / "umap_coords.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"[UMAP] Using cluster column: {cluster_col}")
    print(f"[UMAP] Saved: {out_path}")
    return output


def _wage_to_bubble_size(wage: float, min_size: int = 12, max_size: int = 40) -> int:
    """Scale bubble size to wage (for Plotly marker.size)."""
    wage_min, wage_max = 40_000, 150_000
    clamped = max(wage_min, min(wage_max, wage))
    normalized = (clamped - wage_min) / (wage_max - wage_min)
    return round(min_size + normalized * (max_size - min_size))


if __name__ == "__main__":
    result = generate_umap_coords()
    print("\nSample points:")
    for p in result["points"][:4]:
        print(
            f"  {p['occ_title'][:35]:35s}  x={p['x']:7.3f}  y={p['y']:7.3f}  "
            f"wage=${p['median_wage_me']:>9,.0f}  tier={p['wealth_tier']}  "
            f"cluster={p['cluster_id']}"
        )