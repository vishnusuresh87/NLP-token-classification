"""
evaluation_and_analysis.py
evaluation and error analysis of trained NER model
"""

import pickle
import os
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from config import *
from data_utils import load_plod_dataset
from model_utils import load_model, get_test_predictions
from visualization_utils import (
    print_test_results,
    plot_confusion_matrix,
    extract_entities,
)
from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification


print("EVALUATION & ANALYSIS")

# ============================================================================
# Load preprocessed data and trained model
# ============================================================================

print("\nLoading data and model...")

# Load tokenized datasets
try:
    with open('./preprocessed_data/tokenized_datasets.pkl', 'rb') as f:
        tokenized_datasets = pickle.load(f)
    print("Tokenized datasets loaded")
except FileNotFoundError:
    print("ERROR: Tokenized datasets not found!")
    print("Please run: python data_preprocessing.py")
    exit(1)

# Load trained model and tokenizer
try:
    model, tokenizer = load_model(MODEL_SAVE_DIR)
    print("Model and tokenizer loaded")
except Exception as e:
    print(f"ERROR: Could not load trained model from {MODEL_SAVE_DIR}")
    print(f"Error: {e}")
    print("Please run: python model_training.py")
    exit(1)

# Load original dataset for visualization
dataset = load_plod_dataset()

# Get test dataset
test_dataset = tokenized_datasets['test']
print(f"Test samples: {len(test_dataset)}")

# ============================================================================
# Get predictions on test set
# ============================================================================

print("GETTING TEST PREDICTIONS")

# Create trainer for predictions
data_collator = DataCollatorForTokenClassification(tokenizer)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_eval_batch_size=BATCH_SIZE_EVAL,
    use_cpu=(not torch.cuda.is_available()),
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    processing_class=tokenizer,
)

print("\nRunning predictions on test set...")
predictions, labels = get_test_predictions(trainer, test_dataset)
print(f"Predictions shape: {predictions.shape}")
print(f"Labels shape: {labels.shape}")

# ============================================================================
# Print detailed test results
# ============================================================================

print("TEST SET PERFORMANCE")

print_test_results(predictions, labels)

# ============================================================================
# Create confusion matrix
# ============================================================================

print("GENERATING VISUALIZATIONS")

print("\nCreating confusion matrix...")
# Convert predictions to class IDs if needed
if len(predictions.shape) == 3:
    pred_ids = np.argmax(predictions, axis=2)
else:
    pred_ids = predictions

plot_confusion_matrix(pred_ids, labels)

# ============================================================================
# Error analysis - Find examples with low performance
# ============================================================================

print("ERROR ANALYSIS")

print("\nAnalyzing prediction errors...")

# Convert to label names for analysis
true_predictions = []
true_labels = []

for prediction, label in zip(pred_ids, labels):
    for pred_id, label_id in zip(prediction, label):
        if label_id != -100:
            pred_id = int(pred_id)
            label_id = int(label_id)
            
            if pred_id < len(LABEL_NAMES) and label_id < len(LABEL_NAMES):
                true_predictions.append(LABEL_NAMES[pred_id])
                true_labels.append(LABEL_NAMES[label_id])

# Calculate per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    true_labels, 
    true_predictions, 
    labels=LABEL_NAMES
)

print("\nPer-class performance:")
print(f"{'Label':<10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
for label, p, r, f, s in zip(LABEL_NAMES, precision, recall, f1, support):
    print(f"{label:<10} {p:>10.4f} {r:>10.4f} {f:>10.4f} {s:>10d}")

# ============================================================================
# Sample predictions
# ============================================================================

print("SAMPLE PREDICTIONS")

print("\nShowing first 5 test examples with predictions:\n")

# Get original test data
test_data = dataset['test']

for i in range(min(5, len(test_data))):
    example = test_data[i]
    tokens = example['tokens']
    true_tags = example['ner_tags']
    
    # Get predictions for this example
    pred_ids_example = pred_ids[i]
    
    # Get only valid predictions (non-padding)
    valid_length = len(tokens)
    pred_labels = [LABEL_NAMES[int(pred_ids_example[j])] 
                   for j in range(min(valid_length, len(pred_ids_example)))]
    
    print(f"Example {i+1}:")
    print(f"  Tokens: {' '.join(tokens[:20])}{'...' if len(tokens) > 20 else ''}")
    print(f"  True:   {' '.join(true_tags[:20])}{'...' if len(true_tags) > 20 else ''}")
    print(f"  Pred:   {' '.join(pred_labels[:20])}{'...' if len(pred_labels) > 20 else ''}")
    print()

# ============================================================================
# Summary
# ============================================================================

print("EVALUATION COMPLETE!")
print(f"\nVisualizations saved to: {PLOTS_DIR}")
print(f"  - Confusion matrix: {PLOTS_DIR}/confusion_matrix.png")
