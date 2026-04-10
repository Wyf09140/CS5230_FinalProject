# Maine Construction Career Wealth Navigator — Backend

AI-powered career pathway recommendations for Maine construction workers.
Workers input their current job (or describe their skills in plain text),
choose a mode, and receive ranked pathways with skill gaps, salary uplift,
ROI, Maine training programs, and local employer links.

---

## Project Structure

```
maine_career_backend/
├── data/
│   ├── occupations_master.csv       # 21 Maine construction occupations
│   │                                #   soc_code, occ_title, role_description,
│   │                                #   median_wage_me, wage_source, skill_description
│   ├── training_programs.csv        # 51 real Maine training programs (AGC, SMCC,
│   │                                #   CMCC, EMCC, UMaine, MCA, workforce grants)
│   │                                #   cols: soc_code, occ_title, program_name,
│   │                                #         college, program_type, credential, url, source
│   └── employers.csv                # 1,103 Maine employers + job boards
│                                    #   cols: soc_code, occ_title, employer_name,
│                                    #         agc_type, url, url_type, source_type
│
├── pipeline/
│   ├── embed_and_cluster.py         # SBERT embed → wage blend → 4 clustering methods
│   ├── gap_analysis.py              # Recommendation engine (dynamic method selection)
│   ├── generate_umap.py             # 2D Wealth Galaxy coordinates for frontend
│   └── topic_modeling.py           # LDA + NMF skill topic enrichment (optional)
│
├── models/                          # Auto-generated — do NOT edit manually
│   ├── embeddings.npy               # Raw SBERT vectors (21 × 384)
│   ├── feature_matrix.npy           # Wage-blended feature vectors (21 × 384)
│   ├── kmeans_model.pkl
│   ├── gmm_model.pkl                # Stores soft probabilities for pivot scoring
│   ├── hierarchical_model.pkl       # dict: {labels, linkage_matrix, n_clusters}
│   ├── hdbscan_model.pkl
│   ├── knn_model.pkl
│   ├── occupations_clustered.csv    # All 4 cluster label columns per occupation
│   ├── occupations_embedded.pkl     # Full DataFrame with embedding vectors
│   ├── clustering_meta.json         # ← Per-method silhouette scores + mode hints
│   ├── umap_coords.json             # 2D x/y for Wealth Galaxy scatter plot
│   ├── wealth_tier_map.json
│   ├── pipeline_meta.json           # Backward-compat metadata
│   ├── lda_model.pkl                # Topic model (optional)
│   ├── nmf_model.pkl                # Topic model (optional)
│   ├── occupation_topics.json       # Top-3 topics per occupation
│   └── topic_meta.json
│
├── api/
│   └── main.py                      # FastAPI REST API (10 endpoints)
│
├── run.py                           # Single entry-point for all commands
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on HDBSCAN:** installs Cython extensions — may take 1–2 minutes.
> If it fails on Windows, run `pip install hdbscan --no-build-isolation` or
> use WSL2.

### 3. Run the embedding + clustering pipeline

```bash
python run.py pipeline
```

This is the only step that needs internet access (downloads SBERT model,
~90 MB, cached to `~/.cache/huggingface/` after the first run).
It runs all four clustering methods and writes everything to `models/`.
Takes ~2–4 minutes on first run (SBERT download + clustering).

### 4. Generate Wealth Galaxy 2D coordinates

```bash
python run.py umap
```

Outputs `models/umap_coords.json` — consumed by the frontend Plotly scatter.

### 5. (Optional) Run skill topic modeling

```bash
python run.py topics
```

Generates `models/occupation_topics.json` and `models/topic_meta.json`.
After running, open `models/topic_labels.json` and edit the topic names
based on the top words printed in the terminal.

### 6. Start the API server

```bash
python run.py api
```

Server starts at **http://localhost:8000**.
Interactive Swagger docs at **http://localhost:8000/docs**.

### 7. Smoke test

```bash
python run.py test
```

Runs all recommendation modes against the live engine and prints results.

### Full first-time setup (single command)

```bash
python run.py pipeline umap topics api
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API index + endpoint list |
| `GET` | `/health` | Engine status + all 4 silhouette scores |
| `GET` | `/occupations` | All 21 occupations with all 4 cluster labels |
| `GET` | `/occupations/{soc_code}` | Single occupation — full detail |
| `POST` | `/recommend` | SOC-based pathway (progress / pivot / field_change) |
| `POST` | `/recommend-from-skills` | Free-text skills, no SOC code needed |
| `POST` | `/find-occupation` | Fuzzy-match description to closest SOC codes |
| `GET` | `/wealth-tiers` | Cluster overview — all 4 methods side-by-side |
| `GET` | `/clustering-report` | Full per-method diagnostics + per-occupation labels |
| `GET` | `/training/{soc_code}` | Maine training programs for an occupation |
| `GET` | `/employers/{soc_code}` | Maine employers for an occupation |

---

## Recommendation Modes

### `progress` — Same trade, move up (KMeans)

Worker stays in their trade but advances to a higher-paying role.
KMeans provides clean centroid-based cluster boundaries; same-cluster
occupations get a 10% score bonus.

```json
POST /recommend
{
  "soc_code": "47-2031",
  "mode": "progress",
  "top_n": 3,
  "user_current_wage": 52000
}
```

### `pivot` — Cross to a different cluster (GMM)

Worker moves to a different trade cluster with transferable skills.
GMM soft probabilities identify occupations that sit near cluster
boundaries (high entropy = naturally reachable from multiple starting
points). Score = 50% skill similarity + 30% wage uplift + 20% GMM
boundary score.

```json
POST /recommend
{
  "soc_code": "47-2031",
  "mode": "pivot",
  "top_n": 3,
  "user_current_wage": 52000
}
```

### `field_change` — Any occupation by skill overlap (Agglomerative)

All occupations ranked by semantic skill similarity; no wage filter,
no cluster filter. Agglomerative (hierarchical) labels add a small bonus
for occupations in the same trade family branch. The frontend can apply
its own wage floor slider on top.

```json
POST /recommend
{
  "soc_code": "47-2031",
  "mode": "field_change",
  "top_n": 5
}
```

### Free-text skills — No SOC code (HDBSCAN)

User describes their skills in plain English. SBERT embeds the text and
scores all 21 occupations by cosine similarity. HDBSCAN natural groupings
orient the user's position in the Wealth Galaxy.

```json
POST /recommend-from-skills
{
  "skills_text": "I weld structural steel, read blueprints, and have OSHA-30",
  "top_n": 3,
  "user_current_wage": 48000,
  "target_min_wage": 65000
}
```

---

## Clustering Architecture

Four methods run on the same SBERT+wage feature matrix:

| Method | Best For | Auto-K | Key Advantage |
|--------|----------|--------|---------------|
| KMeans | `progress` | 3–5 Silhouette | Clean centroid ladder |
| GMM | `pivot` | 3–5 BIC | Soft boundary scores |
| Agglomerative | `field_change` | 3–5 Silhouette | Dendrogram / trade families |
| HDBSCAN | `skill_match` | Auto (density) | Natural groupings, noise detection |

`models/clustering_meta.json` stores the Silhouette score for each method
and the `best_method_hints` dict. At request time, `select_best_clustering_method()`
does a dictionary lookup — zero ML inference per request.

Feature matrix formula:
```
feature_vector = 0.65 × SBERT_vector + 0.35 × wage_vector
```
Both components are L2-normalised before blending. Wage weight of 0.35
keeps semantic skill proximity primary while ensuring Wealth Tier clusters
separate meaningfully by pay band.

---

## Data Sources

| File | Source | Size |
|------|--------|------|
| `occupations_master.csv` | O*NET (skills) + Maine DOL/BLS OES (wages) | 21 occupations |
| `training_programs.csv` | AGC Maine, SMCC, CMCC, EMCC, UMaine, USM, MCA, workforce grants | 51 programs |
| `employers.csv` | AGC Maine member directory + job boards (Indeed, AGC, ConstructionJobs) | 1,103 rows |

### Data notes
- `training_programs.csv` — `source_type` ∈ `agc_apprenticeship | community_college | university | workforce_initiative`
- `employers.csv` — `url_type` ∈ `company_site | indeed_fallback | job_board`. `source_type` ∈ `local_employer | job_board`
- Wages use Maine DOL data where available; suppressed codes fall back to NE neighbor average then national.

---

## Quarterly Data Refresh

1. Update `data/occupations_master.csv` with new Maine DOL wage data
2. Re-run:
   ```bash
   python run.py pipeline umap topics
   ```
3. Restart the API server — models lazy-load on first request

---

## Environment Variables (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | API server port |
| `MODELS_DIR` | `models` | Path to model artifacts |
| `DATA_DIR` | `data` | Path to CSV data files |

---

## Frontend Integration Checklist

- **Wealth Galaxy** → `GET /wealth-tiers` (all 4 clustering views) + `models/umap_coords.json`
- **Job dropdown** → `GET /occupations`
- **Free-text matcher** → `POST /find-occupation` → then `POST /recommend`
- **"Change field" flow** → `POST /recommend-from-skills` (no SOC needed)
- **Pathway cards** → `POST /recommend` response: `pathways[].roi`, `skills_to_acquire`, `training_programs`, `maine_employers`
- **Employer badges** → `url_type == "company_site"` → direct link; `"indeed_fallback"` → Indeed search
- **Clustering toggle** → `GET /wealth-tiers` → `clustering_views.{method}` for KMeans/GMM/Agglom/HDBSCAN view
- **Admin/debug** → `GET /clustering-report`

