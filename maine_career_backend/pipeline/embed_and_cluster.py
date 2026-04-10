"""
Maine Career Wealth Navigator — Embedding & Multi-Method Clustering Pipeline
Layer 1 (Deepest Layer): Coefficient Matrices & Wealth Embedding Engine

Runs four clustering algorithms on the same SBERT+wage feature matrix:
  • KMeans          — fast, centroid-based, hard assignments
  • GMM             — soft probabilistic assignments, handles overlapping clusters
  • Agglomerative   — hierarchical, captures nested trade families
  • HDBSCAN         — density-based, finds natural groupings, marks noise

Each is evaluated by Silhouette score.  Results are written to clustering_meta.json
so the API can select the best method per use-case at runtime without re-running
the pipeline.

Artifacts written to models/
─────────────────────────────
  embeddings.npy              raw SBERT vectors (N × 384)p
  feature_matrix.npy          blended embedding+wage vectors (N × dim)
  kmeans_model.pkl
  gmm_model.pkl
  hierarchical_model.pkl      dict: {"linkage_matrix": ..., "labels": ...}
  hdbscan_model.pkl
  knn_model.pkl               NearestNeighbors on feature_matrix (for gap queries)
  occupations_clustered.csv   all four label columns, human-readable
  occupations_embedded.pkl    full DataFrame including embedding/feature vectors
  clustering_meta.json        per-method silhouette scores + best-method hints
  wealth_tier_map.json        SOC → display tier (wage-threshold based, not cluster)
  pipeline_meta.json          embedding info, wage blend weight, tier labels

Usage
─────
  python pipeline/embed_and_cluster.py               # defaults
  python pipeline/embed_and_cluster.py data/my.csv   # custom data
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage as scipy_linkage
import hdbscan
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SBERT_MODEL      = "all-MiniLM-L6-v2"   # 384-dim, ~90 MB download, cached
K_MIN, K_MAX     = 3, 5                  # Auto-K search range (stable for N=21)
WAGE_BLEND       = 0.35                  # Wage signal weight in feature matrix
HDBSCAN_MIN_SAMP = 2                     # Must be low for small dataset (N=21)

# ─────────────────────────────────────────────────────────────────────────────
# Wealth-tier display labels  (wage-threshold based, cluster-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

TIER_THRESHOLDS = [
    (55_000,  "Tier 1 — Entry/Support"),
    (80_000,  "Tier 2 — Technical Core"),
    (110_000, "Tier 3 — Senior/Specialist"),
    (None,    "Tier 4 — Leadership/Expert"),
]
TIER_WAGE_LABELS = {
    "Tier 1 — Entry/Support":     "< $55,000",
    "Tier 2 — Technical Core":    "$55,000 – $79,999",
    "Tier 3 — Senior/Specialist": "$80,000 – $109,999",
    "Tier 4 — Leadership/Expert": "$110,000+",
}

def wage_to_tier(wage: float) -> str:
    for ceiling, label in TIER_THRESHOLDS:
        if ceiling is None or wage < ceiling:
            return label
    return "Tier 4 — Leadership/Expert"

def assign_wealth_tiers(wages: np.ndarray) -> list:
    return [wage_to_tier(w) for w in wages]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — SBERT embeddings
# ─────────────────────────────────────────────────────────────────────────────

def get_embeddings(texts: list) -> np.ndarray:
    """
    Encode skill descriptions with SBERT (all-MiniLM-L6-v2).
    Returns (N, 384) float32, L2-normalised so cosine sim = dot product.
    Model downloads once (~90 MB) and is cached to ~/.cache/huggingface/.
    """
    print(f"  Loading SBERT: {SBERT_MODEL}")
    model = SentenceTransformer(SBERT_MODEL)
    vecs = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  Embedding shape: {vecs.shape}")
    return vecs


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Wage-blended feature matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(embeddings: np.ndarray, wages: np.ndarray,
                          blend: float = WAGE_BLEND) -> np.ndarray:
    """
    Blend L2-normalised SBERT vectors with a wage signal.

    Feature = (1-blend)*skill_vec + blend*wage_vec
    where wage_vec is the Min-Max scaled wage repeated across all dimensions
    then re-normalised so both components contribute equally per-dimension.

    blend=0.35 keeps semantic skill proximity primary while still ensuring
    the Wealth Galaxy clusters separate meaningfully by pay band.
    """
    wage_scaled = MinMaxScaler().fit_transform(wages.reshape(-1, 1))   # 0-1
    wage_feat   = np.repeat(wage_scaled, embeddings.shape[1], axis=1)
    w_norms     = np.linalg.norm(wage_feat, axis=1, keepdims=True)
    wage_feat   = wage_feat / np.where(w_norms == 0, 1.0, w_norms)

    fm    = (1 - blend) * embeddings + blend * wage_feat
    norms = np.linalg.norm(fm, axis=1, keepdims=True)
    return fm / np.where(norms == 0, 1.0, norms)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Cluster label remapping (ascending wage order)
# ─────────────────────────────────────────────────────────────────────────────

def remap_labels_by_wage(labels: np.ndarray, wages: np.ndarray) -> np.ndarray:
    """
    Renumber cluster IDs so cluster 0 always has the lowest median wage.
    This makes cluster IDs interpretable as a Wealth Tier ladder regardless
    of which algorithm assigned them.
    Noise points (label == -1 from HDBSCAN) are kept as -1.
    """
    unique = [c for c in np.unique(labels) if c != -1]
    med    = {c: np.median(wages[labels == c]) for c in unique}
    order  = sorted(med, key=med.get)
    remap  = {old: new for new, old in enumerate(order)}
    out    = np.array([remap.get(l, -1) for l in labels])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Run all four clustering methods
# ─────────────────────────────────────────────────────────────────────────────

def _safe_silhouette(fm: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette on non-noise points only; returns -1.0 if too few clusters."""
    mask = labels != -1
    if mask.sum() < 4 or len(np.unique(labels[mask])) < 2:
        return -1.0
    try:
        return round(float(silhouette_score(fm[mask], labels[mask])), 4)
    except Exception:
        return -1.0


def run_kmeans(fm: np.ndarray, wages: np.ndarray) -> tuple:
    """
    KMeans with auto-K selection (Silhouette over K_MIN..K_MAX).
    Returns (model, labels, silhouette, n_clusters).
    """
    best_k, best_sil = K_MIN, -1.0
    print("  KMeans sweep:")
    for k in range(K_MIN, K_MAX + 1):
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(fm)
        s   = _safe_silhouette(fm, lbl)
        print(f"    K={k}  sil={s:.4f}")
        if s > best_sil:
            best_k, best_sil = k, s

    km = KMeans(n_clusters=best_k, random_state=42, n_init=15)
    raw_labels = km.fit_predict(fm)
    labels     = remap_labels_by_wage(raw_labels, wages)
    # Reorder centroids to match remapped cluster IDs
    order = sorted({c: np.median(wages[raw_labels == c]) for c in range(best_k)},
                   key=lambda c: np.median(wages[raw_labels == c]))
    km.cluster_centers_ = km.cluster_centers_[order]
    sil = _safe_silhouette(fm, labels)
    print(f"  → KMeans best K={best_k}  sil={sil:.4f}")
    return km, labels, sil, best_k


def run_gmm(fm: np.ndarray, wages: np.ndarray) -> tuple:
    """
    Gaussian Mixture Model with auto-component selection via BIC (prefer lower)
    and tie-broken by Silhouette.  Soft probabilities stored on the model object.
    Returns (model, labels, silhouette, n_components).
    """
    best_k, best_bic, best_sil = K_MIN, np.inf, -1.0
    print("  GMM sweep:")
    for k in range(K_MIN, K_MAX + 1):
        try:
            g   = GaussianMixture(n_components=k, covariance_type="full",
                                  random_state=42, n_init=5)
            g.fit(fm)
            lbl = g.predict(fm)
            bic = g.bic(fm)
            s   = _safe_silhouette(fm, lbl)
            print(f"    K={k}  BIC={bic:.1f}  sil={s:.4f}")
            # Prefer lower BIC; use silhouette to break ties within 1% BIC difference
            if bic < best_bic * 0.99 or (abs(bic - best_bic) / max(abs(best_bic), 1) < 0.01
                                          and s > best_sil):
                best_k, best_bic, best_sil = k, bic, s
        except Exception as e:
            print(f"    K={k}  FAILED: {e}")

    gmm = GaussianMixture(n_components=best_k, covariance_type="full",
                          random_state=42, n_init=10)
    gmm.fit(fm)
    raw_labels     = gmm.predict(fm)
    labels         = remap_labels_by_wage(raw_labels, wages)
    gmm.proba_     = gmm.predict_proba(fm)   # store soft probs (N × K)
    sil = _safe_silhouette(fm, labels)
    print(f"  → GMM best K={best_k}  BIC={best_bic:.1f}  sil={sil:.4f}")
    return gmm, labels, sil, best_k


def run_agglomerative(fm: np.ndarray, wages: np.ndarray) -> tuple:
    """
    Agglomerative (hierarchical) clustering with Ward linkage, auto-K.
    Also saves the full scipy linkage matrix for dendrogram rendering.
    Returns (artifact_dict, labels, silhouette, n_clusters).
    """
    best_k, best_sil = K_MIN, -1.0
    print("  Agglomerative sweep:")
    for k in range(K_MIN, K_MAX + 1):
        ag  = AgglomerativeClustering(n_clusters=k, linkage="ward")
        lbl = ag.fit_predict(fm)
        s   = _safe_silhouette(fm, lbl)
        print(f"    K={k}  sil={s:.4f}")
        if s > best_sil:
            best_k, best_sil = k, s

    ag     = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    raw_labels = ag.fit_predict(fm)
    labels     = remap_labels_by_wage(raw_labels, wages)
    linkage_mat = scipy_linkage(fm, method="ward")
    artifact   = {
        "n_clusters":    best_k,
        "labels":        labels,
        "linkage_matrix": linkage_mat,   # for dendrogram in frontend
    }
    sil = _safe_silhouette(fm, labels)
    print(f"  → Agglomerative best K={best_k}  sil={sil:.4f}")
    return artifact, labels, sil, best_k


def run_hdbscan(fm: np.ndarray, wages: np.ndarray) -> tuple:
    """
    HDBSCAN — density-based, discovers K automatically, marks low-density
    occupations as noise (-1).  Works well for finding natural trade families
    that don't have clear centroid structure.
    Returns (model, labels, silhouette, n_clusters_found).
    """
    clf = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=HDBSCAN_MIN_SAMP,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    raw_labels = clf.fit_predict(fm)
    labels     = remap_labels_by_wage(raw_labels, wages)
    n_found    = int((labels != -1).any()) * len(np.unique(labels[labels != -1]))
    n_noise    = int((labels == -1).sum())
    sil        = _safe_silhouette(fm, labels)
    print(f"  → HDBSCAN: {n_found} clusters, {n_noise} noise points, sil={sil:.4f}")
    clf.labels_remapped_ = labels   # store remapped labels on model
    return clf, labels, sil, n_found


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Determine best-method hints
# ─────────────────────────────────────────────────────────────────────────────

def compute_best_method_hints(scores: dict) -> dict:
    """
    Given {method: silhouette_score}, return a dict of use-case hints.

    Design rationale
    ────────────────
    progress   → KMeans preferred: clean centroids make "climb within cluster"
                 intuitive and the cluster_bonus scoring is centroid-aligned.
    pivot      → GMM preferred: soft probabilities expose occupations that sit
                 near cluster boundaries — exactly the lateral pivot candidates.
    field_change → Agglomerative preferred: captures nested trade families
                   so the dendrogram reveals "easy hops" within a branch.
    skill_match  → HDBSCAN preferred when sil > 0 (natural density groupings
                   surface the most coherent skill clusters); falls back to
                   best overall if HDBSCAN found < 2 clusters.
    overall    → simply the highest-silhouette method.
    """
    # Exclude HDBSCAN from "overall best" if it scored negatively
    valid = {m: s for m, s in scores.items() if s > 0}
    overall = max(valid, key=valid.get) if valid else "kmeans"

    hdbscan_ok = scores.get("hdbscan", -1) > 0
    return {
        "overall":      overall,
        "progress":     "kmeans",
        "pivot":        "gmm"    if scores.get("gmm", -1) > 0 else "kmeans",
        "field_change": "agglomerative" if scores.get("agglomerative", -1) > 0 else "kmeans",
        "skill_match":  "hdbscan" if hdbscan_ok else overall,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    data_path="data/occupations_master.csv",
    output_dir="models",
    wage_blend_weight=WAGE_BLEND,
):
    """
    Full pipeline:
      Load → SBERT embed → wage blend → run KMeans/GMM/Agglomerative/HDBSCAN
      → evaluate → save all artifacts + clustering_meta.json.

    The three-layer architecture is preserved:
      Deepest  — four coefficient matrices (one per method) + KNN
      Middle   — this pipeline script (runs offline, quarterly refresh)
      Upper    — FastAPI loads clustering_meta.json and picks the right model
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    df = pd.read_csv(data_path)
    for col in ("skill_description", "median_wage_me", "soc_code", "occ_title"):
        assert col in df.columns, f"Missing column: {col!r}"
    print(f"    {len(df)} occupations")

    # ── 2. SBERT embeddings ───────────────────────────────────────────────────
    print("\n[2] SBERT embeddings...")
    embeddings = get_embeddings(df["skill_description"].tolist())

    # ── 3. Feature matrix ─────────────────────────────────────────────────────
    print(f"\n[3] Building feature matrix (wage_blend={wage_blend_weight})...")
    wages = df["median_wage_me"].values.astype(float)
    feature_matrix = build_feature_matrix(embeddings, wages, blend=wage_blend_weight)
    print(f"    Feature matrix shape: {feature_matrix.shape}")

    # ── 4. Run all clustering methods ─────────────────────────────────────────
    print("\n[4] Running clustering methods...")

    print("\n  ── KMeans ──")
    km_model,   km_labels,   km_sil,   km_k   = run_kmeans(feature_matrix, wages)

    print("\n  ── GMM ──")
    gmm_model,  gmm_labels,  gmm_sil,  gmm_k  = run_gmm(feature_matrix, wages)

    print("\n  ── Agglomerative ──")
    agg_artifact, agg_labels, agg_sil, agg_k  = run_agglomerative(feature_matrix, wages)

    print("\n  ── HDBSCAN ──")
    hdb_model,  hdb_labels,  hdb_sil,  hdb_k  = run_hdbscan(feature_matrix, wages)

    scores = {
        "kmeans":        km_sil,
        "gmm":           gmm_sil,
        "agglomerative": agg_sil,
        "hdbscan":       hdb_sil,
    }

    # ── 5. Enrich DataFrame ───────────────────────────────────────────────────
    print("\n[5] Enriching DataFrame...")
    df["wealth_tier"]        = assign_wealth_tiers(wages)
    df["cluster_kmeans"]     = km_labels
    df["cluster_gmm"]        = gmm_labels
    df["cluster_agglom"]     = agg_labels
    df["cluster_hdbscan"]    = hdb_labels
    df["embedding"]          = list(embeddings)
    df["feature_vector"]     = list(feature_matrix)

    # ── 6. KNN on feature matrix ──────────────────────────────────────────────
    print("\n[6] Fitting KNN (cosine, k=6)...")
    knn = NearestNeighbors(n_neighbors=min(6, len(df)), metric="cosine")
    knn.fit(feature_matrix)

    # ── 7. Save artifacts ─────────────────────────────────────────────────────
    print("\n[7] Saving artifacts...")

    np.save(f"{output_dir}/embeddings.npy", embeddings);       _log(f"{output_dir}/embeddings.npy")
    np.save(f"{output_dir}/feature_matrix.npy", feature_matrix); _log(f"{output_dir}/feature_matrix.npy")

    _pkl(output_dir, "kmeans_model",      km_model)
    _pkl(output_dir, "gmm_model",         gmm_model)
    _pkl(output_dir, "hierarchical_model", agg_artifact)
    _pkl(output_dir, "hdbscan_model",     hdb_model)
    _pkl(output_dir, "knn_model",         knn)

    # occupations_clustered.csv — all label columns, no vectors
    df.drop(columns=["embedding", "feature_vector"]).to_csv(
        f"{output_dir}/occupations_clustered.csv", index=False
    )
    _log(f"{output_dir}/occupations_clustered.csv")

    # occupations_embedded.pkl — full DataFrame with vectors (for inference)
    with open(f"{output_dir}/occupations_embedded.pkl", "wb") as fh:
        pickle.dump(df, fh)
    _log(f"{output_dir}/occupations_embedded.pkl")

    # wealth_tier_map.json — SOC → display tier (wage-threshold, cluster-agnostic)
    tier_map = {
        row["soc_code"]: {
            "occ_title":      row["occ_title"],
            "wealth_tier":    row["wealth_tier"],
            "median_wage_me": row["median_wage_me"],
            "cluster_kmeans":  int(row["cluster_kmeans"]),
            "cluster_gmm":     int(row["cluster_gmm"]),
            "cluster_agglom":  int(row["cluster_agglom"]),
            "cluster_hdbscan": int(row["cluster_hdbscan"]),
        }
        for _, row in df.iterrows()
    }
    with open(f"{output_dir}/wealth_tier_map.json", "w") as fh:
        json.dump(tier_map, fh, indent=2)
    _log(f"{output_dir}/wealth_tier_map.json")

    # clustering_meta.json — the key new artifact
    hints = compute_best_method_hints(scores)
    clustering_meta = {
        "embedding_model":   f"SBERT ({SBERT_MODEL})",
        "embedding_dim":     int(embeddings.shape[1]),
        "wage_blend_weight": wage_blend_weight,
        "n_occupations":     len(df),
        "methods": {
            "kmeans":        {"silhouette": km_sil,  "n_clusters": km_k,  "model_file": "kmeans_model.pkl"},
            "gmm":           {"silhouette": gmm_sil, "n_clusters": gmm_k, "model_file": "gmm_model.pkl"},
            "agglomerative": {"silhouette": agg_sil, "n_clusters": agg_k, "model_file": "hierarchical_model.pkl"},
            "hdbscan":       {"silhouette": hdb_sil, "n_clusters": hdb_k, "model_file": "hdbscan_model.pkl"},
        },
        "best_method_hints": hints,
        "silhouette_ranking": sorted(scores, key=scores.get, reverse=True),
    }
    with open(f"{output_dir}/clustering_meta.json", "w") as fh:
        json.dump(clustering_meta, fh, indent=2)
    _log(f"{output_dir}/clustering_meta.json")

    # pipeline_meta.json — kept for backward compatibility
    pipeline_meta = {
        "k_clusters":        km_k,
        "silhouette_score":  km_sil,
        "embedding_model":   f"SBERT ({SBERT_MODEL})",
        "embedding_dim":     int(embeddings.shape[1]),
        "wage_blend_weight": wage_blend_weight,
        "n_occupations":     len(df),
        "tier_wage_labels":  TIER_WAGE_LABELS,
        "cluster_summary": {
            str(c): {
                "n_occupations": int((km_labels == c).sum()),
                "median_wage":   round(float(np.median(wages[km_labels == c])), 0),
            }
            for c in np.unique(km_labels) if c != -1
        },
    }
    with open(f"{output_dir}/pipeline_meta.json", "w") as fh:
        json.dump(pipeline_meta, fh, indent=2)
    _log(f"{output_dir}/pipeline_meta.json")

    # ── 8. Summary ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PIPELINE COMPLETE — Silhouette Scores")
    print("="*60)
    for method, sil in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * max(0, int(sil * 40)) if sil > 0 else "(noise/invalid)"
        print(f"  {method:<15} {sil:+.4f}  {bar}")
    print(f"\n  Best-method hints: {hints}")
    print(f"\n  KMeans cluster summary:")
    for c in sorted(np.unique(km_labels)):
        if c == -1: continue
        mask   = km_labels == c
        titles = df.loc[mask, "occ_title"].tolist()
        med_w  = np.median(wages[mask])
        print(f"    Cluster {c} | ${med_w:,.0f} | {titles}")

    return df, {
        "kmeans":   km_model,
        "gmm":      gmm_model,
        "agglom":   agg_artifact,
        "hdbscan":  hdb_model,
    }, knn, feature_matrix, clustering_meta


def _pkl(output_dir, name, obj):
    path = f"{output_dir}/{name}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    _log(path)

def _log(path: str):
    print(f"    Saved: {path}")


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/occupations_master.csv"
    run_pipeline(data_path=data_path)
