# -*- coding: utf-8 -*-
"""
Train Mori Technical (FLAN-T5) using the unified training pipeline.
"""
# =============================================================================
# Imports
# =============================================================================
import pandas as pd

from Scripts.config import DATA_PATH
from Scripts.Mori_TechnicalPrompts import build_prompt_training
from Scripts.Mori_TechnicalTokenizer import make_splits, tokenize_datasets
from Scripts.Mori_TechnicalTrainer import train_model
from Scripts.Mori_TechnicalDatasetGeneration import generate_mori_knowledge_dataset

# =============================================================================
# Functions
# =============================================================================

# Load the training dataset and build source/target text fields for Seq2Seq training
def load_and_prepare_df() -> pd.DataFrame:
    """
    Load the Parquet dataset and prepare it for Seq2Seq training.

    The function filters invalid rows and creates:
      - source_text: prompt constructed from each row (input/context/question type)
      - target_text: expected model output

    Returns:
        A pandas DataFrame containing the prepared dataset.
    """
    df = pd.read_parquet(str(DATA_PATH))
    df = df.dropna(subset=["input", "output", "question_type"])

    df["source_text"] = df.apply(build_prompt_training, axis=1)
    df["target_text"] = df["output"]
    return df


def finetune_mori_technical(run_generation: bool = True, mode: str = "early") -> None:
    """
    Execute the end-to-end training workflow for Mori Technical.

    Steps:
      1) (Optional) Generate the base knowledge dataset
      2) Load and prepare the base knowledge dataset
      3) Split into train/validation sets
      4) Tokenize both splits
      5) Train the model
    """
    if run_generation:
        generate_mori_knowledge_dataset()

    df = load_and_prepare_df()

    train_ds, val_ds = make_splits(df)
    tokenizer, train_tok, val_tok = tokenize_datasets(train_ds, val_ds)

    # mode: "normal" or "early"
    train_model(tokenizer, train_tok, val_tok, case=mode)

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    finetune_mori_technical(run_generation=True, mode="early")


# =============================================================================
# Fin
# =============================================================================
