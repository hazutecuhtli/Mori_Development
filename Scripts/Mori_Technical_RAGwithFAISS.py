# -*- coding: utf-8 -*-
"""
Generates a FAISS vector database for Mori using the E5 multilingual embedding model.
Fixed, argument-free version adapted to the latest Mori release.
"""

# =============================================================================
# Imports
# =============================================================================
import json
import os
import warnings
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

#Import config as a package module (no sys.path hacks)
from Scripts import config


# ************************************************************************
# Default configuration
# ************************************************************************
parquet_filename = config.DATA_PATH          # already a Path
RAG_BASE_DIR     = config.VECDB_PATH         # already a Path
model_name       = config.MODEL_E5_NAME
batch_size       = config.VEC_BATCH_SIZE

dedup_cosine_th  = None     # e.g., 0.985 to deduplicate (or None to disable)
version          = "mori-v1.0"
index_basename   = "mori"


# ************************************************************************
# Setup
# ************************************************************************
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running Device:", DEVICE)

RAG_INDEX_PATH = RAG_BASE_DIR / f"{index_basename}.faiss"
RAG_METAS_PATH = RAG_BASE_DIR / f"{index_basename}_metas.json"

# Lazy loading (only first time retrieve_docs is called)
_rag_model = None
_rag_index = None
_rag_metas = None


#=====================================================================================
# Helpers
#=====================================================================================

def load_parquet(path: Path):
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def make_id(r: dict) -> str:
    if "id" in r and r["id"]:
        return str(r["id"])

    term = str(r.get("canonical_term", "")).strip().replace("|", " ")
    ctx  = str(r.get("context", "")).strip().replace("|", " ")
    inp  = str(r.get("input", "")).strip().replace("|", " ")

    head = inp[:40].replace(" ", "_")
    base = f"{term}__{ctx}__{head}".strip("_")

    return base if base else f"rec_{abs(hash(json.dumps(r, ensure_ascii=False))) % (10**12)}"


def build_doc_text_e5(r: dict) -> str:
    term  = r.get("canonical_term", "")
    ctx   = r.get("context", "")
    inp   = r.get("input", "")
    out   = r.get("output", "")
    qtype = r.get("question_type", "")

    body = " | ".join([s for s in [term, ctx, qtype, inp, out] if s])
    return f"passage: {body}"


def batch_encode(model, texts, batch_size: int = 32, normalize: bool = True) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    if emb.dtype != np.float32:
        emb = emb.astype("float32")
    return emb


def build_index(emb: np.ndarray, use_ip: bool = True):
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d) if use_ip else faiss.IndexFlatL2(d)
    index.add(emb)
    return index


def cosine_dedup_mask(emb: np.ndarray, th: float) -> np.ndarray:
    n = emb.shape[0]
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep[i]:
            continue
        sims = (emb[i:i + 1] @ emb[i + 1:].T).ravel()
        dup_idxs = np.where(sims >= th)[0] + (i + 1)
        keep[dup_idxs] = False

    return keep


#=====================================================================================
# RAG retrieval (optional usage)
#=====================================================================================

def _ensure_rag_loaded(verbose: bool = True) -> None:
    global _rag_model, _rag_index, _rag_metas

    if _rag_model is None:
        if verbose:
            print(f"Loading RAG embedding model '{model_name}' on device '{DEVICE}'...")
        _rag_model = SentenceTransformer(model_name, device=DEVICE)

    if _rag_index is None:
        if verbose:
            print(f"Loading FAISS index from '{RAG_INDEX_PATH}'...")
        _rag_index = faiss.read_index(str(RAG_INDEX_PATH))

    if _rag_metas is None:
        if verbose:
            print(f"Loading RAG metadata from '{RAG_METAS_PATH}'...")
        with open(RAG_METAS_PATH, "r", encoding="utf-8") as f:
            _rag_metas = json.load(f)


def retrieve_docs(query: str, k: int = 3, verbose: bool = True):
    _ensure_rag_loaded(verbose)

    qtext = f"query: {query}"
    q_emb = _rag_model.encode(
        [qtext],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, idxs = _rag_index.search(q_emb, k)

    results = []
    for s, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        m = _rag_metas[i]
        results.append({"score": float(s), **m})

    return results


#=====================================================================================
# Build FAISS DB
#=====================================================================================

def generate_rag_faiss_db() -> None:
    RAG_BASE_DIR.mkdir(parents=True, exist_ok=True)

    faiss_path = RAG_BASE_DIR / f"{index_basename}.faiss"
    metas_path = RAG_BASE_DIR / f"{index_basename}_metas.json"
    ids_path   = RAG_BASE_DIR / f"{index_basename}_ids.npy"

    if not parquet_filename.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_filename}")

    data = load_parquet(parquet_filename)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Empty Parquet file or unexpected dataset format.")

    docs, metas, ids = [], [], []
    for r in data:
        rec_id = make_id(r)
        ids.append(rec_id)

        metas.append({
            "id": rec_id,
            "canonical_term": r.get("canonical_term", ""),
            "context": r.get("context", ""),
            "input": r.get("input", ""),
            "output": r.get("output", ""),
            "question_type": r.get("question_type", ""),
            "version": version,
            "encoder": model_name,
        })
        docs.append(build_doc_text_e5(r))

    if len(docs) == 0:
        raise ValueError("No documents were generated from the input dataset.")

    os.environ.setdefault("HF_HUB_TIMEOUT", "60")
    model = SentenceTransformer(model_name, device=DEVICE)

    emb = batch_encode(model, docs, batch_size=batch_size, normalize=True)

    if dedup_cosine_th is not None:
        keep_mask = cosine_dedup_mask(emb, dedup_cosine_th)
        emb   = emb[keep_mask]
        metas = [m for m, k in zip(metas, keep_mask) if k]
        ids   = [rid for rid, k in zip(ids, keep_mask) if k]

    if not (len(emb) == len(metas) == len(ids)):
        raise RuntimeError("Misalignment detected between embeddings, metadata, and record IDs.")

    index = build_index(emb, use_ip=True)
    faiss.write_index(index, str(faiss_path))
    save_json(metas, metas_path)
    np.save(str(ids_path), np.array(ids, dtype=object))

    print(f"*** FAISS index saved to: {faiss_path}")
    print(f"*** Metas saved to:      {metas_path}")
    print(f"*** IDs saved to:        {ids_path}")



# =============================================================================
# Main
# =============================================================================


def main() -> None:
    generate_rag_faiss_db()


if __name__ == "__main__":
    main()


# =============================================================================
# Fin
# =============================================================================
