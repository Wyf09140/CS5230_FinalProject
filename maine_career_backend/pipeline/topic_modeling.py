"""
Maine Career Wealth Navigator - Skill Topic Modeling
Middle Layer: Optional bonus pipeline that enriches occupation profiles
with interpretable skill topics using LDA and NMF.

Why this exists
---------------
SBERT embeddings are semantically rich but opaque - you can't easily
tell *why* two occupations are similar.  Topic modeling adds a human-readable
layer: each occupation gets a distribution over ~8 skill topics like
"Blueprint Reading", "Heavy Equipment", "Project Coordination", etc.

These topics are surfaced in the API as skills_topics per occupation and
used to give more specific gap explanations in /recommend:
  "You need to build skills in: Blueprint Reading, Code Compliance"
instead of just a list of raw keywords.

Outputs written to models/
--------------------------
  lda_model.pkl            Trained LDA model (sklearn)
  nmf_model.pkl            Trained NMF model (sklearn)
  tfidf_for_topics.pkl     TF-IDF vectorizer (topic-specific - NOT SBERT replacement)
  topic_labels.json        {topic_id: human_label} - manually curated after inspection
  occupation_topics.json   {soc_code: {lda_topics: [...], nmf_topics: [...]}}
  topic_meta.json          Model params, top words per topic, coherence proxy

Usage
-----
  python pipeline/topic_modeling.py                  # run both LDA + NMF
  python pipeline/topic_modeling.py --method lda     # LDA only
  python pipeline/topic_modeling.py --method nmf     # NMF only

Note: topic_labels.json must be manually edited after first run.
Run the script, inspect the top words per topic, then assign human labels.
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# -- Config --------------------------------------------------------------------

N_TOPICS       = 8      # Good for 21 occupations: meaningful without over-splitting
N_TOP_WORDS    = 10     # Top words to display per topic (for label assignment)
MAX_FEATURES   = 300    # Vocabulary size for topic TF-IDF (different from SBERT vocab)
NGRAM_RANGE    = (1, 2)
RANDOM_STATE   = 42

# Default topic labels - edit these after inspecting top words per topic
DEFAULT_TOPIC_LABELS = {
    "0": "Heavy Equipment Operation",
    "1": "Blueprint Reading & Planning",
    "2": "Structural Installation",
    "3": "Electrical & Mechanical Systems",
    "4": "Project Management & Coordination",
    "5": "Safety & Compliance",
    "6": "Civil & Site Work",
    "7": "Supervision & Leadership",
}

STOPWORDS = {
    "and", "or", "the", "to", "of", "a", "in", "is", "with", "for", "on",
    "by", "as", "be", "are", "at", "from", "an", "this", "that", "such",
    "each", "have", "has", "may", "not", "using", "use", "used", "other",
    "when", "which", "their", "work", "workers", "working", "tasks", "skills",
}


# -----------------------------------------------------------------------------
# Text preprocessing
# -----------------------------------------------------------------------------

def preprocess_skill_text(text: str) -> str:
    """
    Light preprocessing for topic modeling:
    - Remove section headers (Tasks:, Skills:)
    - Filter short/stop tokens
    Keeps domain-specific compound phrases for n-gram capture.
    """
    text = text.replace("Tasks:", "").replace("Skills:", "")
    tokens = [
        t.lower() for t in text.split()
        if len(t) > 3 and t.lower() not in STOPWORDS and t.replace(",","").isalpha()
    ]
    return " ".join(tokens)


# -----------------------------------------------------------------------------
# LDA (Latent Dirichlet Allocation) - generative probabilistic model
# -----------------------------------------------------------------------------

def run_lda(corpus: list, vocab_matrix) -> tuple:
    """
    Train LDA on count vectors.
    Returns (model, doc_topic_matrix).

    LDA finds topics as probability distributions over words.
    Each occupation gets a probability distribution over N_TOPICS topics.
    Good for: "what proportion of this job is about safety vs. equipment?"
    """
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        random_state=RANDOM_STATE,
        max_iter=100,
        learning_method="online",
        doc_topic_prior=0.1,    # sparse doc-topic distributions (few topics per doc)
        topic_word_prior=0.01,  # sparse topic-word distributions (few words per topic)
    )
    doc_topics = lda.fit_transform(vocab_matrix)
    print(f"  LDA: {N_TOPICS} topics, perplexity={lda.perplexity(vocab_matrix):.1f}")
    return lda, doc_topics


# -----------------------------------------------------------------------------
# NMF (Non-negative Matrix Factorization) - linear algebra decomposition
# -----------------------------------------------------------------------------

def run_nmf(corpus: list, tfidf_matrix) -> tuple:
    """
    Train NMF on TF-IDF vectors.
    Returns (model, doc_topic_matrix).

    NMF tends to produce more coherent, parts-based topics than LDA.
    Good for: discovering additive skill combinations ("this job = 60% planning + 40% equipment").
    """
    nmf = NMF(
        n_components=N_TOPICS,
        random_state=RANDOM_STATE,
        init="nndsvda",          # deterministic init - better convergence on small datasets
        max_iter=500,
        alpha_W=0.1,             # L1 regularisation -> sparser doc representations
        l1_ratio=0.5,
    )
    doc_topics = nmf.fit_transform(tfidf_matrix)
    # Normalise rows to probability distributions
    row_sums = doc_topics.sum(axis=1, keepdims=True)
    doc_topics = doc_topics / np.where(row_sums == 0, 1.0, row_sums)
    print(f"  NMF: {N_TOPICS} topics, reconstruction_err={nmf.reconstruction_err_:.2f}")
    return nmf, doc_topics


# -----------------------------------------------------------------------------
# Extract top words per topic
# -----------------------------------------------------------------------------

def get_top_words(model, feature_names: list, n: int = N_TOP_WORDS) -> dict:
    """Return {topic_id: [top_n_words]} for display and manual labeling."""
    result = {}
    for idx, topic_vec in enumerate(model.components_):
        top_idx = topic_vec.argsort()[::-1][:n]
        result[str(idx)] = [feature_names[i] for i in top_idx]
    return result


# -----------------------------------------------------------------------------
# Build occupation_topics.json
# -----------------------------------------------------------------------------

def build_occupation_topics(
    df: pd.DataFrame,
    lda_doc_topics: np.ndarray,
    nmf_doc_topics: np.ndarray,
    topic_labels: dict,
    top_k: int = 3,
) -> dict:
    """
    For each occupation, store top-K topics from both LDA and NMF.
    top_k=3 gives the UI enough to show meaningful "skill dimensions" per occupation.
    """
    occupation_topics = {}
    for i, (_, row) in enumerate(df.iterrows()):
        soc = row["soc_code"]

        lda_scores = lda_doc_topics[i]
        lda_top    = lda_scores.argsort()[::-1][:top_k]
        lda_result = [
            {"topic_id": int(t), "label": topic_labels.get(str(t), f"Topic {t}"),
             "weight": round(float(lda_scores[t]), 3)}
            for t in lda_top
        ]

        nmf_scores = nmf_doc_topics[i]
        nmf_top    = nmf_scores.argsort()[::-1][:top_k]
        nmf_result = [
            {"topic_id": int(t), "label": topic_labels.get(str(t), f"Topic {t}"),
             "weight": round(float(nmf_scores[t]), 3)}
            for t in nmf_top
        ]

        occupation_topics[soc] = {
            "occ_title":  row["occ_title"],
            "lda_topics": lda_result,
            "nmf_topics": nmf_result,
        }
    return occupation_topics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run_topic_modeling(
    data_path: str = "data/occupations_master.csv",
    output_dir: str = "models",
    method: str = "both",   # "lda", "nmf", or "both"
):
    """
    Full topic modeling pipeline.
    Run after embed_and_cluster.py - this is an enrichment step, not a replacement.
    """
    os.makedirs(output_dir, exist_ok=True)
    p = Path(output_dir)

    print(f"\n[Topic Modeling] method={method}, n_topics={N_TOPICS}")

    # -- Load data -------------------------------------------------------------
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} occupations")

    # -- Preprocess ------------------------------------------------------------
    corpus = [preprocess_skill_text(t) for t in df["skill_description"]]

    # -- Vectorize for LDA (count vectors) ------------------------------------
    count_vec = CountVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        stop_words=list(STOPWORDS),
    )
    count_matrix = count_vec.fit_transform(corpus)

    # -- Vectorize for NMF (TF-IDF vectors) -----------------------------------
    tfidf_vec = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        stop_words=list(STOPWORDS),
        sublinear_tf=True,
    )
    tfidf_matrix = tfidf_vec.fit_transform(corpus)

    # Save this vectorizer separately - NOT the same as SBERT's feature space
    with open(p / "tfidf_for_topics.pkl", "wb") as fh:
        pickle.dump(tfidf_vec, fh)
    print("  Saved: tfidf_for_topics.pkl")

    feature_names = tfidf_vec.get_feature_names_out().tolist()

    # -- Run models ------------------------------------------------------------
    lda_model = lda_doc_topics = None
    nmf_model = nmf_doc_topics = None

    if method in ("lda", "both"):
        print("\n  Running LDA...")
        lda_model, lda_doc_topics = run_lda(corpus, count_matrix)
        with open(p / "lda_model.pkl", "wb") as fh:
            pickle.dump(lda_model, fh)
        print("  Saved: lda_model.pkl")

    if method in ("nmf", "both"):
        print("\n  Running NMF...")
        nmf_model, nmf_doc_topics = run_nmf(corpus, tfidf_matrix)
        with open(p / "nmf_model.pkl", "wb") as fh:
            pickle.dump(nmf_model, fh)
        print("  Saved: nmf_model.pkl")

    # -- Load or generate topic labels -----------------------------------------
    labels_path = p / "topic_labels.json"
    if labels_path.exists():
        with open(labels_path) as fh:
            topic_labels = json.load(fh)
        print(f"  Loaded existing topic_labels.json")
    else:
        topic_labels = DEFAULT_TOPIC_LABELS.copy()
        with open(labels_path, "w") as fh:
            json.dump(topic_labels, fh, indent=2)
        print(f"  Created default topic_labels.json - REVIEW AND EDIT after inspecting top words!")

    # -- Top words per topic ---------------------------------------------------
    lda_top_words = get_top_words(lda_model, feature_names) if lda_model else {}
    nmf_top_words = get_top_words(nmf_model, feature_names) if nmf_model else {}

    # -- Occupation topics -----------------------------------------------------
    if lda_doc_topics is not None and nmf_doc_topics is not None:
        occ_topics = build_occupation_topics(df, lda_doc_topics, nmf_doc_topics, topic_labels)
        with open(p / "occupation_topics.json", "w", encoding="utf-8") as fh:
            json.dump(occ_topics, fh, indent=2)
        print("  Saved: occupation_topics.json")

    # -- Topic meta ------------------------------------------------------------
    topic_meta = {
        "n_topics":       N_TOPICS,
        "n_top_words":    N_TOP_WORDS,
        "max_features":   MAX_FEATURES,
        "ngram_range":    list(NGRAM_RANGE),
        "method":         method,
        "topic_labels":   topic_labels,
        "lda_top_words":  lda_top_words,
        "nmf_top_words":  nmf_top_words,
    }
    with open(p / "topic_meta.json", "w", encoding="utf-8") as fh:
        json.dump(topic_meta, fh, indent=2)
    print("  Saved: topic_meta.json")

    # -- Print topic summary for manual label review ----------------------------
    print("\n" + "="*60)
    print("  TOPIC SUMMARY - edit topic_labels.json with human labels")
    print("="*60)
    active_top_words = nmf_top_words if nmf_top_words else lda_top_words
    for tid, words in active_top_words.items():
        label = topic_labels.get(tid, f"Topic {tid}")
        print(f"  Topic {tid} [{label}]: {', '.join(words[:6])}")

    print("\n  Topic modeling complete.")
    return lda_model, nmf_model, topic_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skill Topic Modeling (LDA + NMF)")
    parser.add_argument("--data", default="data/occupations_master.csv")
    parser.add_argument("--output", default="models")
    parser.add_argument("--method", choices=["lda", "nmf", "both"], default="both")
    args = parser.parse_args()
    run_topic_modeling(data_path=args.data, output_dir=args.output, method=args.method)
