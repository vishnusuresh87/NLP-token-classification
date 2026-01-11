"""
data_preprocessing.py
Loads dataset, tokenizes, computes class weights, and saves all outputs
"""

import pickle
import os
from config import *
from data_utils import (
    load_plod_dataset,
    load_tokenizer,
    preprocess_dataset,
    get_class_weights,
)
from visualization_utils import print_class_weights


print("DATA PREPROCESSING")

# ============================================================================
# Load dataset and tokenizer
# ============================================================================

print("\nLoading dataset and tokenizer...")
dataset = load_plod_dataset()
tokenizer = load_tokenizer()
print("Dataset and tokenizer loaded")

print(f"Dataset: {DATASET_NAME}")
print(f"Model: {MODEL_NAME}")
print(f"Splits: {list(dataset.keys())}")

# ============================================================================
# Tokenize and align labels
# ============================================================================

print("\nTokenizing and aligning labels...")
tokenized_datasets = preprocess_dataset(dataset, tokenizer)
print("Tokenization complete")

print(f"Tokenized splits: {list(tokenized_datasets.keys())}")
for split in tokenized_datasets.keys():
    print(f"    - {split}: {len(tokenized_datasets[split])} samples")

# ============================================================================
# Compute class weights
# ============================================================================

print("\nComputing class weights...")
class_weights = get_class_weights(dataset)
print_class_weights(class_weights)

# ============================================================================
# Save all outputs
# ============================================================================

print("SAVING PREPROCESSED DATA")

# Create output directory
os.makedirs('./preprocessed_data', exist_ok=True)

# Save tokenized datasets
print("\nSaving tokenized datasets...")
with open('./preprocessed_data/tokenized_datasets.pkl', 'wb') as f:
    pickle.dump(tokenized_datasets, f)
print("Saved to: ./preprocessed_data/tokenized_datasets.pkl")

# Save class weights
print("\nSaving class weights...")
with open('./preprocessed_data/class_weights.pkl', 'wb') as f:
    pickle.dump(class_weights, f)
print("Saved to: ./preprocessed_data/class_weights.pkl")

# Save tokenizer
print("\nSaving tokenizer...")
tokenizer.save_pretrained('./preprocessed_data/tokenizer')
print("Saved to: ./preprocessed_data/tokenizer")

# ============================================================================
# Summary
# ============================================================================

print("DATA PREPROCESSING COMPLETE!")
print(f"\nAll preprocessed data saved to: ./preprocessed_data/")
