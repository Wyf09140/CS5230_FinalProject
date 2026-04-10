"""
Maine Career Wealth Navigator - Run Script

Usage:
  python run.py pipeline          # SBERT embed + all 4 clustering methods
  python run.py topics            # LDA + NMF skill topic modeling (bonus)
  python run.py umap              # UMAP 2D Wealth Galaxy coordinates
  python run.py api               # Start FastAPI server (port 8000)
  python run.py test              # Smoke-test the recommendation engine
  python run.py pipeline topics umap api   # Full setup then start server
"""

import sys


def run_pipeline():
    print("=" * 62)
    print("  Embedding & Multi-Method Clustering Pipeline")
    print("=" * 62)
    from pipeline.embed_and_cluster import run_pipeline as _run
    _run(data_path="data/occupations_master.csv", output_dir="models")


def run_topics():
    print("=" * 62)
    print("  Skill Topic Modeling (LDA + NMF)")
    print("=" * 62)
    from pipeline.topic_modeling import run_topic_modeling
    run_topic_modeling(data_path="data/occupations_master.csv", output_dir="models")


def run_umap():
    print("=" * 62)
    print("  UMAP 2D Wealth Galaxy Coordinates")
    print("=" * 62)
    from pipeline.generate_umap import generate_umap_coords
    generate_umap_coords(models_dir="models", data_dir="data")


def run_api():
    print("=" * 62)
    print("  FastAPI Server  ->  http://localhost:8000")
    print("  Swagger docs    ->  http://localhost:8000/docs")
    print("=" * 62)
    import uvicorn
    # reload=False on Windows avoids the multiprocessing re-spawn issue.
    # Use reload=True only if you need hot-reload during active development
    # and are running on Linux/macOS or inside WSL2.
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)


def run_test():
    print("=" * 62)
    print("  Smoke Test - Recommendation Engine")
    print("=" * 62)
    from pipeline.gap_analysis import RecommendationEngine, select_best_clustering_method

    engine = RecommendationEngine()

    print("\n[1] select_best_clustering_method per mode:")
    for mode in ["progress", "pivot", "field_change", "skill_match"]:
        m   = select_best_clustering_method(mode, engine.clustering_meta)
        sil = engine.clustering_meta["methods"][m]["silhouette"]
        print(f"  {mode:<15} -> {m}  (sil={sil:.4f})")

    tests = [
        ("47-2031", "progress",     None,  "Carpenter -> progress (KMeans)"),
        ("47-2031", "pivot",       52000,  "Carpenter -> pivot (GMM, wage=$52k)"),
        ("47-4051", "field_change", None,  "Highway Maint -> field_change (Agglom)"),
    ]
    for soc, mode, wage, label in tests:
        print(f"\n[{label}]")
        r = engine.recommend(soc, mode=mode, top_n=3, user_current_wage=wage)
        q = r["query"]
        print(f"  {q['occ_title']} | ${q['effective_wage']:,.0f} | method={q['clustering_method']}")
        for p in r["pathways"]:
            print(f"  -> {p['target_title']:<45} ${p['target_median_wage']:>9,.0f}  "
                  f"gap={p['skill_gap_pct']}%  5yr=${p['roi']['five_year_net_gain']:,.0f}")

    print("\n[recommend_from_skills]")
    r = engine.recommend_from_skills(
        "I operate excavators and cranes, read site plans, and have OSHA-30",
        top_n=3, user_current_wage=50000, target_min_wage=65000
    )
    q = r["query"]
    print(f"  method={q['clustering_method']} | closest={q['closest_soc_match']['occ_title']}")
    for p in r["pathways"]:
        print(f"  -> {p['target_title']:<45} ${p['target_median_wage']:>9,.0f}  "
              f"breakeven={p['roi']['breakeven_months']} mo")

    print("\n[clustering_report]")
    report = engine.clustering_report()
    print(f"  Silhouette ranking: {report['silhouette_ranking']}")

    print("\nAll tests passed.")


# Windows requires the if __name__ guard so that spawned worker processes
# (used by joblib inside HDBSCAN/umap-learn) do not re-execute this script.
if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)
    if "pipeline" in args:
        run_pipeline()
    if "topics" in args:
        run_topics()
    if "umap" in args:
        run_umap()
    if "test" in args:
        run_test()
    if "api" in args:
        run_api()   # blocking - must be last
