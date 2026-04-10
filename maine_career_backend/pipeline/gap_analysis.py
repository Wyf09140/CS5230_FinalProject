"""
Maine Career Wealth Navigator - Gap Analysis & Recommendation Engine
Layer 1 (continued): Shortest Bridge to Higher Pay

Recommendation modes
  progress      Move up within the same trade cluster (same craft, higher creds).
                Uses KMeans labels - clean centroid boundaries.
  pivot         Lateral move to a different cluster with transferable skills.
                Uses GMM soft-probability labels - surfaces boundary occupations.
  field_change  All occupations ranked by semantic skill similarity to current SOC.
                Uses Agglomerative labels - reveals nested trade family structure.
  (from skills) No SOC code: free-text skill input, HDBSCAN natural groupings.

Dynamic clustering method selection
  select_best_clustering_method(mode, clustering_meta) reads clustering_meta.json
  and returns the method name that performed best for that use-case.
  The engine then loads that method's labels from the DataFrame, enabling
  the gap analysis and scoring to operate on the optimal cluster structure
  for each request - without re-running any ML.

User wage input
  All modes accept optional user_current_wage.
  When provided it overrides the SOC median in ROI calculations,
  giving breakeven_months and five_year_net_gain specific to each worker.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal, Optional
from sentence_transformers import SentenceTransformer

SBERT_MODEL = "all-MiniLM-L6-v2"

MODE_CLUSTER_COL = {
    "progress":     "cluster_kmeans",
    "pivot":        "cluster_gmm",
    "field_change": "cluster_agglom",
}


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def select_best_clustering_method(mode: str, clustering_meta: dict) -> str:
    """
    Read clustering_meta and return the method best suited to the given mode.
    Priority:
      1. Pre-computed hint from pipeline (best_method_hints[mode]).
      2. Fall back to overall best if the hinted method has sil <= 0.
      3. Fall back to "kmeans" as the guaranteed-stable baseline.
    """
    hints   = clustering_meta.get("best_method_hints", {})
    methods = clustering_meta.get("methods", {})

    key    = mode if mode in hints else "overall"
    hinted = hints.get(key, "kmeans")

    if methods.get(hinted, {}).get("silhouette", -1) > 0:
        return hinted

    overall = hints.get("overall", "kmeans")
    if methods.get(overall, {}).get("silhouette", -1) > 0:
        return overall

    return "kmeans"


# ---------------------------------------------------------------------------
# Recommendation Engine
# ---------------------------------------------------------------------------

class RecommendationEngine:
    """
    Singleton-safe inference engine.
    Loads all model artifacts once at startup (via FastAPI lifespan).
    All four clustering models are available in memory; the active label
    column is selected per-request based on mode and clustering_meta hints.
    """

    def __init__(self, models_dir: str = "models", data_dir: str = "data"):
        self.models_dir = Path(models_dir)
        self.data_dir   = Path(data_dir)
        self._sbert: Optional[SentenceTransformer] = None
        self._load_artifacts()

    # -- Artifact loading ----------------------------------------------------

    def _load_artifacts(self) -> None:
        p = self.models_dir

        with open(p / "occupations_embedded.pkl", "rb") as fh:
            self.df = pickle.load(fh)

        with open(p / "kmeans_model.pkl", "rb") as fh:
            self.kmeans = pickle.load(fh)
        with open(p / "gmm_model.pkl", "rb") as fh:
            self.gmm = pickle.load(fh)
        with open(p / "hierarchical_model.pkl", "rb") as fh:
            self.hierarchical = pickle.load(fh)
        with open(p / "hdbscan_model.pkl", "rb") as fh:
            self.hdbscan_model = pickle.load(fh)

        self.feature_matrix = np.load(str(p / "feature_matrix.npy"))
        self.embeddings     = np.load(str(p / "embeddings.npy"))

        with open(p / "knn_model.pkl", "rb") as fh:
            self.knn = pickle.load(fh)

        with open(p / "wealth_tier_map.json", encoding="utf-8") as fh:
            self.tier_map = json.load(fh)

        with open(p / "clustering_meta.json", encoding="utf-8") as fh:
            self.clustering_meta = json.load(fh)

        with open(p / "pipeline_meta.json", encoding="utf-8") as fh:
            self.meta = json.load(fh)

        self.training_df  = pd.read_csv(self.data_dir / "training_programs.csv")
        self.employers_df = pd.read_csv(self.data_dir / "employers.csv")

        best = self.clustering_meta.get("best_method_hints", {}).get("overall", "kmeans")
        print(
            f"Engine loaded: {len(self.df)} occupations | "
            f"embedding={self.clustering_meta['embedding_model']} | "
            f"best_overall={best}"
        )
        for method, info in self.clustering_meta["methods"].items():
            print(f"  {method:<15} sil={info['silhouette']:+.4f}  K={info['n_clusters']}")

    # -- SBERT (lazy-loaded once) ---------------------------------------------

    def _get_sbert(self) -> SentenceTransformer:
        if self._sbert is None:
            self._sbert = SentenceTransformer(SBERT_MODEL)
        return self._sbert

    def _embed_text(self, text: str) -> np.ndarray:
        vec = self._get_sbert().encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )
        return vec[0]

    # -- Clustering method selection ------------------------------------------

    def _get_cluster_col(self, mode: str) -> str:
        best_method = select_best_clustering_method(mode, self.clustering_meta)
        col_map = {
            "kmeans":        "cluster_kmeans",
            "gmm":           "cluster_gmm",
            "agglomerative": "cluster_agglom",
            "hdbscan":       "cluster_hdbscan",
        }
        return col_map.get(best_method, "cluster_kmeans")

    # -- GMM soft-probability pivot scoring -----------------------------------

    def _gmm_boundary_score(self, src_idx: int) -> np.ndarray:
        """
        Per-occupation boundary score using GMM soft probabilities.
        High entropy = occupation sits between clusters = good pivot candidate.
        Returns (N,) array in [0, 1].
        """
        if not hasattr(self.gmm, "proba_"):
            return np.zeros(len(self.df))
        proba = self.gmm.proba_
        with np.errstate(divide="ignore", invalid="ignore"):
            log_p = np.where(proba > 0, np.log(proba), 0.0)
        entropy     = -np.sum(proba * log_p, axis=1)
        max_entropy = np.log(proba.shape[1])
        return entropy / max_entropy if max_entropy > 0 else entropy

    # -- Maths helpers --------------------------------------------------------

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _skill_gap_pct(self, src_embed: np.ndarray, tgt_idx: int) -> float:
        sim = self._cosine_sim(src_embed, self.embeddings[tgt_idx])
        return round((1 - sim) * 100, 1)

    # -- Keyword gap extraction -----------------------------------------------

    _STOPWORDS = {
        "and", "or", "the", "to", "of", "a", "in", "is", "with",
        "for", "on", "by", "as", "be", "are", "at", "from", "an",
        "this", "that", "such", "each", "have", "has", "may", "not",
        "using", "use", "used", "their", "other", "when", "which",
    }

    def _gap_keywords(self, src_text: str, tgt_soc: str, top_n: int = 6) -> list:
        tgt_row   = self.df[self.df["soc_code"] == tgt_soc].iloc[0]
        src_words = set(src_text.lower().split())
        tgt_words = [
            w for w in tgt_row["skill_description"].lower().split()
            if len(w) > 4 and w.isalpha() and w not in self._STOPWORDS
        ]
        gap = sorted(set(tgt_words) - src_words, key=len, reverse=True)
        return gap[:top_n]

    # -- ROI model ------------------------------------------------------------

    @staticmethod
    def _roi(current_wage: float, target_wage: float,
             training_cost: float = 5_000) -> dict:
        uplift_annual  = target_wage - current_wage
        uplift_monthly = uplift_annual / 12
        breakeven = round(training_cost / uplift_monthly, 1) if uplift_monthly > 0 else None
        return {
            "wage_used_for_roi":      round(current_wage, 0),
            "annual_salary_uplift":   round(uplift_annual, 0),
            "monthly_uplift":         round(uplift_monthly, 0),
            "training_cost_estimate": training_cost,
            "breakeven_months":       breakeven,
            "five_year_net_gain":     round(uplift_annual * 5 - training_cost, 0),
        }

    # -- Support data ---------------------------------------------------------
    # Column schemas match the real uploaded CSVs:
    #   training_programs.csv: soc_code, occ_title, program_name, college,
    #                          program_type, credential, url, source
    #   employers.csv:         soc_code, occ_title, employer_name, agc_type,
    #                          url, url_type, source_type

    def _get_training(self, soc_code: str) -> list:
        rows = self.training_df[self.training_df["soc_code"] == soc_code]
        if rows.empty:
            return []
        keep = [c for c in ["program_name", "college", "program_type",
                             "credential", "url", "source"] if c in rows.columns]
        return rows[keep].to_dict("records")

    def _get_employers(self, soc_code: str) -> list:
        rows = self.employers_df[self.employers_df["soc_code"] == soc_code]
        if rows.empty:
            return []
        keep = [c for c in ["employer_name", "agc_type", "url",
                             "url_type", "source_type"] if c in rows.columns]
        return rows[keep].to_dict("records")

    # -- Pathway builder (shared by all modes) --------------------------------

    def _build_pathways(
        self,
        src_embed: np.ndarray,
        src_text: str,
        effective_wage: float,
        candidates: pd.DataFrame,
        top_n: int,
        active_method: str = "kmeans",
    ) -> list:
        top = candidates.nlargest(top_n, "_score")
        pathways = []
        for rank, (tgt_idx, tgt_row) in enumerate(top.iterrows(), start=1):
            tgt_soc  = tgt_row["soc_code"]
            gap_pct  = self._skill_gap_pct(src_embed, tgt_idx)
            gap_kw   = self._gap_keywords(src_text, tgt_soc)
            roi      = self._roi(effective_wage, tgt_row["median_wage_me"])
            pathways.append({
                "rank":                 rank,
                "target_soc":           tgt_soc,
                "target_title":         tgt_row["occ_title"],
                "target_wealth_tier":   tgt_row["wealth_tier"],
                "target_median_wage":   tgt_row["median_wage_me"],
                "skill_similarity_pct": round(100 - gap_pct, 1),
                "skill_gap_pct":        gap_pct,
                "skills_to_acquire":    gap_kw,
                "roi":                  roi,
                "training_programs":    self._get_training(tgt_soc),
                "maine_employers":      self._get_employers(tgt_soc),
                "cluster_method_used":  active_method,
            })
        return pathways

    # -------------------------------------------------------------------------
    # PUBLIC: recommend()  -  SOC-code-based
    # -------------------------------------------------------------------------

    def recommend(
        self,
        soc_code: str,
        mode: Literal["progress", "pivot", "field_change"] = "progress",
        top_n: int = 5,
        user_current_wage: Optional[float] = None,
    ) -> dict:
        """
        Generate ranked career pathways for a worker identified by SOC code.

        mode:
          progress     - same trade, move up (KMeans clusters)
          pivot        - different cluster, transferable skills (GMM soft boundaries)
          field_change - any occupation by skill overlap (Agglomerative branches)
        """
        src_rows = self.df[self.df["soc_code"] == soc_code]
        if src_rows.empty:
            raise ValueError(f"SOC code {soc_code!r} not in dataset.")

        src_row      = src_rows.iloc[0]
        src_idx      = src_rows.index[0]
        src_embed    = self.embeddings[src_idx]
        src_text     = src_row["skill_description"]
        src_wage     = src_row["median_wage_me"]
        effective_wage = user_current_wage or src_wage
        wage_label   = "user-reported" if user_current_wage else "SOC median"

        active_method = select_best_clustering_method(mode, self.clustering_meta)
        cluster_col   = self._get_cluster_col(mode)
        src_cluster   = src_row[cluster_col]

        all_others = self.df[self.df["soc_code"] != soc_code].copy()

        if mode == "progress":
            cands = all_others[all_others["median_wage_me"] > src_wage].copy()
            cands["_sim"] = cands.index.map(
                lambda i: self._cosine_sim(src_embed, self.embeddings[i])
            )
            cands["_cluster_bonus"] = (
                (cands[cluster_col] == src_cluster).astype(float) * 0.10
            )
            cands["_score"] = cands["_sim"] + cands["_cluster_bonus"]

        elif mode == "pivot":
            cands = all_others[
                (all_others["median_wage_me"] > src_wage) &
                (all_others[cluster_col] != src_cluster)
            ].copy()
            if cands.empty:
                raise ValueError(
                    f"No pivot candidates for {soc_code!r} - all higher-wage "
                    f"occupations are in the same {active_method} cluster."
                )
            boundary_scores = self._gmm_boundary_score(src_idx)
            cands["_sim"]      = cands.index.map(
                lambda i: self._cosine_sim(src_embed, self.embeddings[i])
            )
            cands["_boundary"] = cands.index.map(
                lambda i: float(boundary_scores[i])
            )
            max_wage = cands["median_wage_me"].max()
            cands["_wage_score"] = cands["median_wage_me"] / max_wage
            cands["_score"] = (
                0.50 * cands["_sim"]
                + 0.30 * cands["_wage_score"]
                + 0.20 * cands["_boundary"]
            )

        else:  # field_change
            cands = all_others.copy()
            cands["_sim"] = cands.index.map(
                lambda i: self._cosine_sim(src_embed, self.embeddings[i])
            )
            cands["_branch_bonus"] = (
                (cands[cluster_col] == src_cluster).astype(float) * 0.05
            )
            cands["_score"] = cands["_sim"] + cands["_branch_bonus"]

        if cands.empty:
            raise ValueError(f"No candidates for soc_code={soc_code!r}, mode={mode!r}.")

        pathways = self._build_pathways(
            src_embed, src_text, effective_wage, cands, top_n, active_method
        )

        return {
            "query": {
                "soc_code":             soc_code,
                "occ_title":            src_row["occ_title"],
                "soc_median_wage":      src_wage,
                "effective_wage":       effective_wage,
                "wage_source":          wage_label,
                "current_wealth_tier":  src_row["wealth_tier"],
                "current_cluster_id":   int(src_cluster) if src_cluster != -1 else -1,
                "mode":                 mode,
                "clustering_method":    active_method,
            },
            "pathways": pathways,
            "metadata": {
                "n_occupations":   len(self.df),
                "embedding_model": self.clustering_meta["embedding_model"],
                "active_method":   active_method,
                "all_silhouettes": {
                    m: v["silhouette"]
                    for m, v in self.clustering_meta["methods"].items()
                },
            },
        }

    # -------------------------------------------------------------------------
    # PUBLIC: recommend_from_skills()  -  free-text, no SOC needed
    # -------------------------------------------------------------------------

    def recommend_from_skills(
        self,
        skills_text: str,
        top_n: int = 5,
        user_current_wage: Optional[float] = None,
        target_min_wage: Optional[float] = None,
    ) -> dict:
        """
        Field-change path driven entirely by free-text skill description.
        HDBSCAN natural groupings used to orient the user in the Wealth Galaxy.
        No SOC code required.
        """
        if not skills_text or not skills_text.strip():
            raise ValueError("skills_text must be a non-empty string.")

        user_vec       = self._embed_text(skills_text)
        DEFAULT_WAGE   = 45_000.0
        effective_wage = user_current_wage or DEFAULT_WAGE
        wage_label     = "user-reported" if user_current_wage else "default ($45,000)"

        active_method  = select_best_clustering_method("skill_match", self.clustering_meta)

        cands = self.df.copy()
        cands["_sim"] = cands.index.map(
            lambda i: self._cosine_sim(user_vec, self.embeddings[i])
        )

        if target_min_wage is not None:
            cands = cands[cands["median_wage_me"] >= target_min_wage]
            if cands.empty:
                raise ValueError(
                    f"No occupations at or above ${target_min_wage:,.0f}."
                )

        cands["_score"] = cands["_sim"]
        top_match = cands.nlargest(1, "_score").iloc[0]

        pathways = self._build_pathways(
            user_vec, skills_text, effective_wage, cands, top_n, active_method
        )

        return {
            "query": {
                "mode":                "field_change_from_skills",
                "skills_text_preview": skills_text[:200],
                "effective_wage":      effective_wage,
                "wage_source":         wage_label,
                "target_min_wage":     target_min_wage,
                "clustering_method":   active_method,
                "closest_soc_match": {
                    "soc_code":   top_match["soc_code"],
                    "occ_title":  top_match["occ_title"],
                    "similarity": round(float(top_match["_sim"]) * 100, 1),
                },
            },
            "pathways": pathways,
            "metadata": {
                "n_occupations":   len(self.df),
                "embedding_model": self.clustering_meta["embedding_model"],
                "active_method":   active_method,
                "all_silhouettes": {
                    m: v["silhouette"]
                    for m, v in self.clustering_meta["methods"].items()
                },
            },
        }

    # -------------------------------------------------------------------------
    # PUBLIC: find_closest_occupation()
    # -------------------------------------------------------------------------

    def find_closest_occupation(self, user_text: str, top_k: int = 3) -> list:
        """Fuzzy-match free-text to the closest SOC codes by SBERT cosine similarity."""
        user_vec = self._embed_text(user_text)
        sims = [self._cosine_sim(user_vec, self.embeddings[i])
                for i in range(len(self.embeddings))]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [
            {
                "soc_code":       self.df.iloc[i]["soc_code"],
                "occ_title":      self.df.iloc[i]["occ_title"],
                "similarity_pct": round(sims[i] * 100, 1),
                "median_wage_me": self.df.iloc[i]["median_wage_me"],
                "wealth_tier":    self.df.iloc[i]["wealth_tier"],
            }
            for i in top_idx
        ]

    # -------------------------------------------------------------------------
    # PUBLIC: clustering_report()
    # -------------------------------------------------------------------------

    def clustering_report(self) -> dict:
        """Full clustering diagnostics for the /clustering-report endpoint."""
        per_occupation = []
        for _, row in self.df.iterrows():
            per_occupation.append({
                "soc_code":        row["soc_code"],
                "occ_title":       row["occ_title"],
                "wealth_tier":     row["wealth_tier"],
                "median_wage_me":  row["median_wage_me"],
                "cluster_kmeans":  int(row["cluster_kmeans"]),
                "cluster_gmm":     int(row["cluster_gmm"]),
                "cluster_agglom":  int(row["cluster_agglom"]),
                "cluster_hdbscan": int(row["cluster_hdbscan"]),
            })
        return {
            "methods":           self.clustering_meta["methods"],
            "best_method_hints": self.clustering_meta["best_method_hints"],
            "silhouette_ranking": self.clustering_meta["silhouette_ranking"],
            "per_occupation":    per_occupation,
        }
