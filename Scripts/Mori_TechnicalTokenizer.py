# -*- coding: utf-8 -*-
"""
Dataset splitting and tokenization utilities for Mori (Seq2Seq training).
"""

# =============================================================================
# Imports
# =============================================================================
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from Scripts.config import MODEL_NAME, MAX_SOURCE_LEN, MAX_TARGET_LEN, TEST_SIZE, RANDOM_SEED

# =============================================================================
# Splitting
# =============================================================================

# Create stratified train/validation splits using question_type as the label
def make_splits(df):
    """
    Create stratified train/validation splits from a prepared pandas DataFrame.

    The function builds a Hugging Face Dataset from the required columns,
    encodes `question_type` as categorical labels, shuffles the data, and
    performs a stratified split to preserve class proportions.

    Args:
        df: Pandas DataFrame containing at least:
            - source_text
            - target_text
            - question_type

    Returns:
        A tuple (train_ds, val_ds) as Hugging Face Datasets.
    """
    hf_ds = Dataset.from_pandas(df[["source_text", "target_text", "question_type"]])

    # Convert question_type into a categorical column suitable for stratified splitting
    hf_ds = hf_ds.class_encode_column("question_type")

    # Shuffle and split while preserving question_type distribution
    hf_ds = hf_ds.shuffle(seed=RANDOM_SEED)
    split: DatasetDict = hf_ds.train_test_split(
        test_size=TEST_SIZE,
        stratify_by_column="question_type",
    )

    return split["train"], split["test"]


# =============================================================================
# Tokenization
# =============================================================================

# Tokenize train/validation datasets for Seq2Seq training (inputs + labels)
def tokenize_datasets(train_ds, val_ds):
    """
    Tokenize the train and validation datasets for Seq2Seq training.

    This function creates a tokenizer from MODEL_NAME and maps a preprocessing
    function over the datasets to produce model-ready fields:
      - input_ids, attention_mask
      - labels (tokenized target_text)

    Args:
        train_ds: Hugging Face Dataset containing source_text and target_text.
        val_ds: Hugging Face Dataset containing source_text and target_text.

    Returns:
        A tuple (tokenizer, train_tok, val_tok) where:
          - tokenizer: Hugging Face tokenizer instance
          - train_tok: tokenized training dataset
          - val_tok: tokenized validation dataset
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess_batch(batch):
        # Tokenize inputs (source_text)
        inputs = tokenizer(
            batch["source_text"],
            max_length=MAX_SOURCE_LEN,
            truncation=True,
            padding="max_length",
        )

        # Tokenize targets (target_text) as labels
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target_text"],
                max_length=MAX_TARGET_LEN,
                truncation=True,
                padding="max_length",
            )["input_ids"]

        inputs["labels"] = labels
        return inputs

    train_tok = train_ds.map(
        preprocess_batch,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    val_tok = val_ds.map(
        preprocess_batch,
        batched=True,
        remove_columns=val_ds.column_names,
    )

    return tokenizer, train_tok, val_tok

# =============================================================================
# Fin
# =============================================================================
