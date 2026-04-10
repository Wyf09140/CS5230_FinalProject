"""
Maine Career Wealth Navigator - FastAPI Backend  v3.0
Upper Layer: REST API consumed by the React/Next.js frontend

New in v3: ensemble clustering
  - Four clustering models per pipeline run (KMeans, GMM, Agglomerative, HDBSCAN)
  - Each request uses the best model for its mode (from clustering_meta.json)
  - /clustering-report exposes all four sets of labels + silhouette scores
  - /wealth-tiers now shows cluster assignments across all methods

Endpoints
---------
  GET  /                          API index
  GET  /health                    Liveness + engine + clustering status
  GET  /occupations               All 21 occupations (dropdown)
  GET  /occupations/{soc_code}    Single occupation detail
  POST /recommend                 SOC-based pathway (progress/pivot/field_change)
  POST /recommend-from-skills     Free-text skill input, no SOC needed
  POST /find-occupation           Fuzzy SOC code lookup
  GET  /wealth-tiers              Cluster overview (all four methods)
  GET  /clustering-report         Full per-method diagnostics
  GET  /training/{soc_code}       Maine training programs
  GET  /employers/{soc_code}      Maine AGC employers

Run
---
  uvicorn api.main:app --reload --port 8000
  # Swagger docs: http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional

# -- Singleton engine ----------------------------------------------------------
_ENGINE = None

def get_engine():
    if _ENGINE is None:
        raise RuntimeError("Engine not initialised - lifespan startup failed.")
    return _ENGINE

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all model artifacts once before accepting requests."""
    global _ENGINE
    from pipeline.gap_analysis import RecommendationEngine
    _ENGINE = RecommendationEngine(models_dir="models", data_dir="data")
    print("Recommendation engine ready.")
    yield


# -- App -----------------------------------------------------------------------
app = FastAPI(
    title="Maine Construction Career Wealth Navigator",
    description=(
        "AI-powered career pathway recommendations for Maine construction workers. "
        "SBERT embeddings + ensemble clustering (KMeans / GMM / Agglomerative / HDBSCAN). "
        "Active clustering method is selected per-request based on mode and "
        "offline Silhouette evaluation."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Request models ------------------------------------------------------------

class RecommendRequest(BaseModel):
    soc_code: str = Field(..., examples=["47-2031"])
    mode: Literal["progress", "pivot", "field_change"] = Field(
        "progress",
        description=(
            "progress      - same trade, move up (KMeans clusters).\n"
            "pivot         - different cluster, transferable skills (GMM soft boundaries).\n"
            "field_change  - any occupation by skill overlap (Agglomerative branches)."
        ),
    )
    top_n: int = Field(3, ge=1, le=5)
    user_current_wage: Optional[float] = Field(
        None, ge=20_000, le=500_000,
        description="Worker's actual annual wage. Overrides SOC median in ROI.",
        examples=[52000],
    )


class RecommendFromSkillsRequest(BaseModel):
    skills_text: str = Field(
        ..., min_length=10,
        examples=["I weld structural steel, read blueprints, and hold an OSHA-10 card."],
        description="Free-text description of current skills. No SOC code needed.",
    )
    top_n: int = Field(3, ge=1, le=5)
    user_current_wage: Optional[float] = Field(
        None, ge=20_000, le=500_000,
        description="Actual wage; defaults to $45,000 if omitted.",
    )
    target_min_wage: Optional[float] = Field(
        None, ge=30_000,
        description="Only return occupations at or above this Maine annual wage.",
        examples=[70000],
    )


class FindOccupationRequest(BaseModel):
    description: str = Field(
        ..., min_length=5,
        examples=["I operate heavy equipment at road construction sites"],
    )
    top_k: int = Field(3, ge=1, le=5)


# -- Routes --------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness + engine status. Returns silhouette scores for all clustering methods."""
    engine = get_engine()
    cm = engine.clustering_meta
    return {
        "status":          "ok",
        "service":         "Maine Career Wealth Navigator v3",
        "occupations":     len(engine.df),
        "embedding_model": cm["embedding_model"],
        "best_overall":    cm["best_method_hints"]["overall"],
        "clustering": {
            m: {"silhouette": v["silhouette"], "n_clusters": v["n_clusters"]}
            for m, v in cm["methods"].items()
        },
    }


@app.get("/occupations")
def list_occupations():
    """All 21 occupations with all four cluster assignments - for the UI dropdown."""
    engine = get_engine()
    cols = [
        "soc_code", "occ_title", "role_description",
        "median_wage_me", "wage_source", "wealth_tier",
        "cluster_kmeans", "cluster_gmm", "cluster_agglom", "cluster_hdbscan",
    ]
    records = engine.df[cols].to_dict("records")
    for r in records:
        for col in ("cluster_kmeans", "cluster_gmm", "cluster_agglom", "cluster_hdbscan"):
            r[col] = int(r[col])
    return {"count": len(records), "occupations": records}


@app.get("/occupations/{soc_code}")
def get_occupation(soc_code: str):
    """Detail for one occupation including all cluster labels, training, and employers."""
    engine = get_engine()
    rows = engine.df[engine.df["soc_code"] == soc_code]
    if rows.empty:
        raise HTTPException(404, detail=f"SOC code {soc_code!r} not found.")
    row = rows.iloc[0]
    return {
        "soc_code":                  row["soc_code"],
        "occ_title":                 row["occ_title"],
        "role_description":          row["role_description"],
        "median_wage_me":            row["median_wage_me"],
        "wage_source":               row["wage_source"],
        "wealth_tier":               row["wealth_tier"],
        "cluster_kmeans":            int(row["cluster_kmeans"]),
        "cluster_gmm":               int(row["cluster_gmm"]),
        "cluster_agglom":            int(row["cluster_agglom"]),
        "cluster_hdbscan":           int(row["cluster_hdbscan"]),
        "skill_description_preview": row["skill_description"][:500] + "...",
        "training_programs":         engine._get_training(soc_code),
        "maine_employers":           engine._get_employers(soc_code),
    }


@app.post("/recommend")
def recommend(req: RecommendRequest):
    """
    Core recommendation endpoint.

    Clustering method is selected automatically per mode:
      progress     -> KMeans   (clean centroid ladder - same trade, move up)
      pivot        -> GMM      (soft-probability boundary scoring)
      field_change -> Agglomerative (nested trade family branches)

    Response includes `clustering_method` and `all_silhouettes` in metadata
    so the frontend can display which model was active.
    """
    engine = get_engine()
    try:
        return engine.recommend(
            soc_code=req.soc_code,
            mode=req.mode,
            top_n=req.top_n,
            user_current_wage=req.user_current_wage,
        )
    except ValueError as exc:
        raise HTTPException(404, detail=str(exc))


@app.post("/recommend-from-skills")
def recommend_from_skills(req: RecommendFromSkillsRequest):
    """
    Field-change recommendation from free-text skills - no SOC code needed.
    Uses HDBSCAN natural groupings when available; falls back to best overall.
    """
    engine = get_engine()
    try:
        return engine.recommend_from_skills(
            skills_text=req.skills_text,
            top_n=req.top_n,
            user_current_wage=req.user_current_wage,
            target_min_wage=req.target_min_wage,
        )
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc))


@app.post("/find-occupation")
def find_occupation(req: FindOccupationRequest):
    """Fuzzy-match free-text job description to closest SOC codes."""
    engine = get_engine()
    return {"matches": engine.find_closest_occupation(req.description, top_k=req.top_k)}


@app.get("/wealth-tiers")
def get_wealth_tiers():
    """
    Cluster overview for Wealth Galaxy visualisation.
    Returns cluster membership for all four methods so the frontend can
    let users toggle between clustering views.
    """
    engine = get_engine()
    cm     = engine.clustering_meta

    def cluster_view(col: str, method: str) -> dict:
        """Build cluster summary for one clustering method."""
        unique = sorted(engine.df[col].unique())
        return {
            str(c): {
                "cluster_id":    int(c),
                "is_noise":      c == -1,
                "n_occupations": int((engine.df[col] == c).sum()),
                "median_wage":   round(float(engine.df.loc[engine.df[col] == c, "median_wage_me"].median()), 0)
                                 if c != -1 else None,
                "occupations":   engine.df.loc[engine.df[col] == c,
                                     ["soc_code", "occ_title", "wealth_tier", "median_wage_me"]
                                 ].to_dict("records"),
            }
            for c in unique
        }

    return {
        "embedding_model":   cm["embedding_model"],
        "tier_wage_labels":  engine.meta.get("tier_wage_labels", {}),
        "best_method_hints": cm["best_method_hints"],
        "clustering_views": {
            "kmeans":        {
                "silhouette": cm["methods"]["kmeans"]["silhouette"],
                "n_clusters": cm["methods"]["kmeans"]["n_clusters"],
                "clusters":   cluster_view("cluster_kmeans", "kmeans"),
            },
            "gmm": {
                "silhouette": cm["methods"]["gmm"]["silhouette"],
                "n_clusters": cm["methods"]["gmm"]["n_clusters"],
                "clusters":   cluster_view("cluster_gmm", "gmm"),
            },
            "agglomerative": {
                "silhouette": cm["methods"]["agglomerative"]["silhouette"],
                "n_clusters": cm["methods"]["agglomerative"]["n_clusters"],
                "clusters":   cluster_view("cluster_agglom", "agglomerative"),
            },
            "hdbscan": {
                "silhouette": cm["methods"]["hdbscan"]["silhouette"],
                "n_clusters": cm["methods"]["hdbscan"]["n_clusters"],
                "clusters":   cluster_view("cluster_hdbscan", "hdbscan"),
            },
        },
    }


@app.get("/clustering-report")
def clustering_report():
    """
    Full clustering diagnostics - per-method silhouette scores,
    best-method hints, and per-occupation cluster assignments across all four methods.
    Designed for the data-science / admin view in the frontend.
    """
    return get_engine().clustering_report()


@app.get("/training/{soc_code}")
def get_training(soc_code: str):
    """
    Maine training programs for an occupation.
    Columns: program_name, college, program_type, credential, url, source.
    """
    engine   = get_engine()
    programs = engine._get_training(soc_code)
    if not programs:
        raise HTTPException(404, detail=f"No training programs for SOC {soc_code!r}.")
    return {"soc_code": soc_code, "count": len(programs), "programs": programs}


@app.get("/employers/{soc_code}")
def get_employers(soc_code: str):
    """
    Maine employers for an occupation.
    Columns: employer_name, agc_type, url, url_type, source_type.
    source_type = 'local_employer' | 'job_board'
    url_type    = 'company_site' | 'indeed_fallback' | 'job_board'
    """
    engine    = get_engine()
    employers = engine._get_employers(soc_code)
    if not employers:
        raise HTTPException(404, detail=f"No employers for SOC {soc_code!r}.")
    return {"soc_code": soc_code, "count": len(employers), "employers": employers}


@app.get("/")
def root():
    return {
        "service": "Maine Construction Career Wealth Navigator API v3",
        "docs":    "/docs",
        "clustering": "Ensemble: KMeans + GMM + Agglomerative + HDBSCAN",
        "endpoints": {
            "GET  /health":                "Engine + clustering status",
            "GET  /occupations":           "All occupations (all 4 cluster columns)",
            "GET  /occupations/{soc}":     "Single occupation detail",
            "POST /recommend":             "SOC-based pathway - mode selects best clustering method",
            "POST /recommend-from-skills": "Free-text skills, no SOC needed",
            "POST /find-occupation":       "Fuzzy SOC lookup",
            "GET  /wealth-tiers":          "Cluster overview - all 4 methods",
            "GET  /clustering-report":     "Full per-method diagnostics",
            "GET  /training/{soc}":        "Maine training programs",
            "GET  /employers/{soc}":       "Maine AGC employers",
        },
    }
