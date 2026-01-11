"""
model_utils.py
Model training and inference utilities for PLOD-CW-25 NER
Compatible with CPU/GPU
Includes partial layer freezing for efficient fine-tuning
"""

import os
import torch
import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import classification_report, f1_score
from config import *


# ---------------------------------------------------------------------------
# Layer freezing
# ---------------------------------------------------------------------------

def freeze_bert_layers(model, num_layers_to_freeze=6):
    """
    Freeze the first N layers of BERT encoder for efficient fine-tuning.
    
    Args:
        model: BertForTokenClassification model
        num_layers_to_freeze: Number of encoder layers to freeze (0-12)
                              Recommended: 6 for BERT-base (50% freezing)
    
    Freezes:
        - All embedding layers (word, position, token-type)
        - First N transformer encoder layers
    
    Trains:
        - Last (12 - N) encoder layers
        - Classification head (new task-specific layer)
    """
    # Freeze embeddings (shared representations)
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    
    # Freeze first N encoder layers
    for layer in model.bert.encoder.layer[:num_layers_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False
    
    print(f"Frozen layers:")
    print(f"    - Embeddings (word, position, token-type)")
    print(f"    - Encoder layers: 0-{num_layers_to_freeze-1} (out of 0-11)")
    print(f"Trainable layers:")
    print(f"    - Encoder layers: {num_layers_to_freeze}-11 (out of 0-11)")
    print(f"    - Classification head (newly initialized)")


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------

def get_training_arguments():
    """Get TrainingArguments for Hugging Face Trainer."""

    # Ensure directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE_TRAIN,
        per_device_eval_batch_size=BATCH_SIZE_EVAL,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        eval_strategy=EVALUATION_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        save_total_limit=SAVE_TOTAL_LIMIT,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        logging_dir=LOG_DIR,
        logging_steps=100,
        seed=SEED,
        push_to_hub=False,
        report_to=[],  # Disable tensorboard to avoid import errors
        use_cpu=(device == "cpu"),  # Use CPU if not CUDA
    )

    return args


# ---------------------------------------------------------------------------
# Model / tokenizer loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(freeze_layers=True, num_layers_to_freeze=6):
    """
    Load pre-trained PubMedBERT model and tokenizer.
    
    Args:
        freeze_layers: If True, freeze first N layers for efficient fine-tuning
        num_layers_to_freeze: Number of encoder layers to freeze (default: 6)
    """

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )

    # Optionally freeze layers
    if freeze_layers:
        print("LAYER FREEZING CONFIGURATION")
        freeze_bert_layers(model, num_layers_to_freeze)

    # Calculate and print parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nModel loaded: {MODEL_NAME}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Tokenizer loaded")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Metrics - FIXED for seqeval (expects list of lists)
# ---------------------------------------------------------------------------

def compute_metrics(eval_preds):
    """
    Compute seqeval F1 for evaluation.
    FIXED: seqeval expects list of lists (sequences), not flat lists.
    """
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=2)

    # Convert to list of lists (seqeval format)
    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
        pred_seq = []
        label_seq = []
        
        for pred_id, label_id in zip(prediction, label):
            if label_id != -100:  # Skip special tokens
                if pred_id < len(LABEL_NAMES):
                    pred_seq.append(LABEL_NAMES[pred_id])
                if label_id < len(LABEL_NAMES):
                    label_seq.append(LABEL_NAMES[label_id])
        
        # Only add sequences that have labels
        if label_seq:
            true_predictions.append(pred_seq)
            true_labels.append(label_seq)

    # seqeval expects: [[label1, label2, ...], [label1, label2, ...], ...], ie list of lists
    return {
        "f1": f1_score(true_labels, true_predictions),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(train_dataset, eval_dataset, model, tokenizer, class_weights=None):
    """Train the model using Hugging Face Trainer."""

    print("\nTraining model...\n")

    training_args = get_training_arguments()
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Optional: class weights can be passed via a custom loss if needed
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer (new API)
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    return trainer, train_result


# ---------------------------------------------------------------------------
# Test prediction helper
# ---------------------------------------------------------------------------

def get_test_predictions(trainer, test_dataset):
    """
    Run the trainer on the test_dataset and return (predictions, labels).

    predictions: np.ndarray of shape (num_samples, seq_len, num_labels)
    labels:      np.ndarray of shape (num_samples, seq_len)
    """
    output = trainer.predict(test_dataset)
    preds = output.predictions
    labels = output.label_ids
    return preds, labels


# ---------------------------------------------------------------------------
# Saving / loading trained model
# ---------------------------------------------------------------------------

def save_model(model, tokenizer, save_dir=None):
    """Save model and tokenizer to disk."""
    if save_dir is None:
        save_dir = MODEL_SAVE_DIR

    os.makedirs(save_dir, exist_ok=True)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"\n Model saved to: {save_dir}")


def load_model(model_dir=None):
    """Load a previously saved model and tokenizer."""
    if model_dir is None:
        model_dir = MODEL_SAVE_DIR

    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print(f"Model loaded from: {model_dir}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference for a single text
# ---------------------------------------------------------------------------

def get_predictions(model, tokenizer, text, device=None):
    """Get token-level predictions for a single input text string."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    encoding = tokenizer(
        text.split(),
        is_split_into_words=True,
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(
            **{k: v.to(device) for k, v in encoding.items() if k != "offset_mapping"}
        )
        logits = outputs.logits

    predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
    word_ids = encoding.word_ids()

    previous_word_idx = None
    results = []

    for word_idx, pred_id in zip(word_ids, predictions):
        if word_idx is None:
            continue
        elif word_idx != previous_word_idx:
            results.append(
                {
                    "token": text.split()[word_idx],
                    "label": LABEL_NAMES[pred_id],
                }
            )
        previous_word_idx = word_idx

    return results


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------

def print_model_summary(model):
    """Print a simple model summary."""
    print("MODEL SUMMARY")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("\nModel architecture:")
    print(model)
