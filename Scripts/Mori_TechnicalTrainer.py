# -*- coding: utf-8 -*-
"""Script to fine tune a FLAN-T5 Model using a data processing concepts dataset."""
#=====================================================================================
# Importing Libraries  ===============================================================
#=====================================================================================
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from Scripts.config import MODEL_NAME, OUTPUT_DIR, TEMPORAL_DIR
from dataclasses import dataclass
from typing import Optional, Literal
import torch

# ************************************************************************
# Seeting scripts local variables
# ************************************************************************
TrainCase = Literal["normal", "early"]

# ************************************************************************
# Defining custom functions
# ************************************************************************

# Configuration container for training hyperparameters

@dataclass(frozen=True)
class TrainerConfig:
    """
    Immutable configuration class for defining training hyperparameters.

    This dataclass groups all training-related parameters in a single,
    type-safe structure. The configuration is immutable to prevent
    accidental changes during execution and to improve reproducibility
    of experimental results.
    """

    tmp_dir: str
    num_train_epochs: int = 40
    per_device_train_batch_size: int = 12
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.06
    save_total_limit: int = 3
    early_stopping_patience: Optional[int] = 3  # Set to None to disable early stopping

    
# Determine whether bf16 training can be safely enabled on the current GPU

### Important Note:
#
# - bfloat16 (bf16) is used to reduce numerical precision during training in order
#   to improve memory efficiency and computational throughput. Unlike float16,
#   bf16 preserves the dynamic range of float32, which helps maintain stable
#   optimization behavior and model performance in practice when supported
#   by the underlying hardware.


def _gpu_supports_bf16() -> bool:
    """
    Check whether the current CUDA GPU supports bfloat16 precision.

    This helper function is used to conditionally enable bf16 training
    only on compatible hardware (NVIDIA Ampere architecture or newer).
    If CUDA is unavailable or the device capability cannot be queried,
    bf16 support is conservatively disabled.
    """
    if not torch.cuda.is_available():
        return False

    try:
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 8  # Returnds capability >= 8.0 (Ampere or newer)
    except RuntimeError:
        # Fallback: disable bf16 if device capability cannot be determined
        return False


# Build a consistent and reproducible set of training arguments for Seq2Seq models

def _build_training_args(cfg: TrainerConfig, use_bf16: bool) -> Seq2SeqTrainingArguments:
    """
    Construct HuggingFace Seq2SeqTrainingArguments from a fixed configuration.

    This helper centralizes all training hyperparameters to ensure consistency
    across experiments. Some options (e.g., early stopping and best-model
    selection) may be overridden at a higher level depending on the training
    strategy.

    Args:
        cfg: Training configuration object containing the hyperparameters.
        use_bf16: Whether bf16 training should be enabled (hardware-dependent).

    Returns:
        A Seq2SeqTrainingArguments instance configured for the current run.
    """
    return Seq2SeqTrainingArguments(
        output_dir=cfg.tmp_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="linear",
        optim="adafactor",              # Recommended optimizer for T5-like models
        weight_decay=0.0,               # Disable regularization to favor memorization
        max_grad_norm=1.0,
        logging_strategy="epoch",
        save_total_limit=cfg.save_total_limit,
        fp16=False,                     # Prefer bf16 when available for stability
        bf16=use_bf16,
        report_to="none",
        disable_tqdm=True,              # Cleaner logs in scripts and notebooks
        load_best_model_at_end=False,   # Enabled only when early stopping is used
    )


# Train a Seq2Seq model using a configurable training strategy (standard or early stopping)

def train_model(tokenizer, train_tok, val_tok, case: TrainCase = "normal") -> None:
    """
    Train a Seq2Seq model (e.g., T5/FLAN-T5) and save the trained artifacts.

    The function supports two training modes:
    - "normal": fixed number of epochs, no early stopping, final model is saved.
    - "early": enables early stopping and saves the best checkpoint according to eval_loss.

    Args:
        tokenizer: Tokenizer compatible with the selected MODEL_NAME.
        train_tok: Tokenized training dataset.
        val_tok: Tokenized validation dataset.
        case: Training mode. Must be "normal" or "early".

    Raises:
        ValueError: If `case` is not one of {"normal", "early"}.
    """
    if case not in ("normal", "early"):
        raise ValueError(f"Invalid case: {case}. Expected 'normal' or 'early'.")

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    if torch.cuda.is_available():
        model = model.to("cuda")

    use_bf16 = _gpu_supports_bf16()

    # Select hyperparameters by training mode
    if case == "normal":
        cfg = TrainerConfig(
            tmp_dir=TEMPORAL_DIR,
            learning_rate=4e-4,
            warmup_ratio=0.01,
            save_total_limit=2,
            early_stopping_patience=None,
        )
        enable_early_stopping = False
    else:
        cfg = TrainerConfig(
            tmp_dir=TEMPORAL_DIR,
            learning_rate=2e-4,
            warmup_ratio=0.06,
            save_total_limit=3,
            early_stopping_patience=3,
        )
        enable_early_stopping = cfg.early_stopping_patience is not None

    args = _build_training_args(cfg, use_bf16)

    callbacks = None
    if enable_early_stopping:
        # Configure best-model selection for early stopping
        args.load_best_model_at_end = True
        args.metric_for_best_model = "eval_loss"
        args.greater_is_better = False
        args.predict_with_generate = True

        callbacks = [EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)]

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    trainer.train()

    # If early stopping is enabled and load_best_model_at_end=True, the trainer already holds the best model here.
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

#=====================================================================================
# Fin
#=====================================================================================

