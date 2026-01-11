"""
dataset_analysis.py
Dataset loading and exploratory analysis
Run this first to understand the PLOD-CW-25 dataset
"""

import sys
from data_utils import load_plod_dataset, analyze_dataset, load_tokenizer
from visualization_utils import (
    print_dataset_stats,
    plot_label_distribution,
    plot_sequence_length_distribution
)
from config import *

print("DATASET ANALYSIS")

print("\nLoading PLOD-CW-25 dataset...")
dataset = load_plod_dataset()
print(f"Dataset loaded successfully")

print("\n Analyzing dataset statistics...")
stats = analyze_dataset(dataset)
print_dataset_stats(stats)

print("\n Loading PubMedBERT tokenizer...")
tokenizer = load_tokenizer()
print(f"Tokenizer loaded: {MODEL_NAME}")
print(f"Vocabulary size: {len(tokenizer)}")

print("\n Creating visualizations...")
plot_label_distribution(dataset)
print(f"Label distribution plot saved to {PLOTS_DIR}/label_distribution.png")

plot_sequence_length_distribution(dataset)
print(f"Sequence length distribution plot saved to {PLOTS_DIR}/sequence_length_distribution.png")


