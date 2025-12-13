# Scripts/config.py
"""
Central configuration file for the Mori project.

This module defines all shared paths, model identifiers, and training
parameters used across:
  - training pipelines
  - RAG (FAISS) construction
  - inference (CLI & Streamlit apps)

All paths are anchored to the repository root to ensure portability
across environments (IDE, CLI, Streamlit, CI, Docker).
"""

from pathlib import Path

# =============================================================================
# Project root
# =============================================================================
# This file lives at: <REPO_ROOT>/Scripts/config.py
# Using __file__ guarantees correct path resolution regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# Model inference configuration
# =============================================================================

# Local directory containing the fine-tuned Mori Technical model
# Used by both terminal and Streamlit applications
LLM1_DIR = PROJECT_ROOT / "Models" / "mori-flan-v3"

# Secondary LLM used for experimentation or future extensions
# (e.g., larger instruction-following models)
# NOTE: This is a Hugging Face model ID, not a local path.
LLM2_DIR = "Qwen/Qwen2-1.5B-Instruct"

# Embedding model used to generate vector representations for RAG
# Chosen for strong multilingual (Spanish/English) semantic performance
MODEL_E5_NAME = "intfloat/multilingual-e5-base"


# =============================================================================
# Dataset paths
# =============================================================================

# Directory containing raw JSON files used to generate Mori's knowledge base
JSONs_PATH = PROJECT_ROOT / "Data"

# Unified Parquet dataset used for both:
#   - LLM fine-tuning
#   - FAISS vector database construction
DATA_PATH = PROJECT_ROOT / "Data" / "Mori_Technical_Final.parquet"


# =============================================================================
# LLM training configuration
# =============================================================================

# Base model used as the starting point for fine-tuning
MODEL_NAME = "google/flan-t5-base"

# Output directory where the fine-tuned model checkpoints are saved
# (e.g., early stopping, best checkpoint)
OUTPUT_DIR = PROJECT_ROOT / "Models" / "mori-flan-v4_early4"

# Temporary directory used to store intermediate training artifacts 
TEMPORAL_DIR = PROJECT_ROOT / "Scripts" / "mori_tecnico_es_tmp"


# =============================================================================
# RAG (Retrieval-Augmented Generation) settings
# =============================================================================

# Directory where FAISS index, metadata, and record IDs are stored
VECDB_PATH = PROJECT_ROOT / "Vec_DataBase"

# Batch size used when generating embeddings for FAISS
# (This does NOT affect LLM training batch size)
VEC_BATCH_SIZE = 32


# =============================================================================
# Training parameters
# =============================================================================

# Random seed for reproducibility across dataset splits and training
RANDOM_SEED = 42

# Fraction of the dataset reserved for validation
TEST_SIZE = 0.1

# Maximum token length for the encoder (input prompt)
MAX_SOURCE_LEN = 64

# Maximum token length for the decoder (expected answer)
MAX_TARGET_LEN = 92
