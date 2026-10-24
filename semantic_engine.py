"""Transformer-based semantic matching for RoleFit AI."""

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource(show_spinner="Loading semantic matching model...")
def load_embedding_model():
    """Load the pretrained sentence-transformer once per app process."""
    return SentenceTransformer(MODEL_NAME)


@st.cache_data(show_spinner="Encoding job descriptions...")
def encode_job_texts(texts):
    """Convert job descriptions into reusable dense embeddings."""
    model = load_embedding_model()
    embeddings = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)


def rank_by_semantic_similarity(profile_text, job_embeddings):
    """Return one semantic similarity score per job, scaled to 0–1."""
    model = load_embedding_model()
    profile_embedding = model.encode(
        [profile_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    scores = cosine_similarity(profile_embedding, job_embeddings)[0]

    # Cosine similarities can be negative; scale to a bounded range.
    scores = np.clip(scores, 0.0, 1.0)
    return scores.astype(float)
