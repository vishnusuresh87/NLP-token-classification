"""
model_training.py
Train NER model using preprocessed data
"""

import pickle
from config import *
from model_utils import (
    load_model_and_tokenizer,
    train_model,
    save_model,
    get_test_predictions,
)
from visualization_utils import (
    print_class_weights,
    print_test_results,
    plot_label_distribution,
    plot_sequence_length_distribution,
)
from data_utils import load_plod_dataset


print("MODEL TRAINING")

# ============================================================================
# Load preprocessed data
# ============================================================================

print("\nLoading preprocessed data...")

# Load tokenized datasets
try:
    with open('./preprocessed_data/tokenized_datasets.pkl', 'rb') as f:
        tokenized_datasets = pickle.load(f)
    print("Tokenized datasets loaded")
except FileNotFoundError:
    print("ERROR: Preprocessed data not found!")
    print("Please run: python data_preprocessing.py")
    exit(1)

# Load class weights
try:
    with open('./preprocessed_data/class_weights.pkl', 'rb') as f:
        class_weights = pickle.load(f)
    print("Class weights loaded")
    print_class_weights(class_weights)
except FileNotFoundError:
    print("ERROR: Class weights not found!")
    print("Please run: python data_preprocessing.py")
    exit(1)

# Split into train/eval/test
train_dataset = tokenized_datasets['train']
eval_dataset = tokenized_datasets['validation']
test_dataset = tokenized_datasets['test']

print(f"\nDataset splits:")
print(f"  Train samples: {len(train_dataset)}")
print(f"  Eval samples: {len(eval_dataset)}")
print(f"  Test samples: {len(test_dataset)}")

# ============================================================================
# Load model and tokenizer
# ============================================================================

print("Loading PubMedBERT model...")
model, tokenizer = load_model_and_tokenizer(freeze_layers=True, num_layers_to_freeze=6)

# ============================================================================
# Train model
# ============================================================================

print("TRAINING MODEL")

trainer, train_result = train_model(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    model=model,
    tokenizer=tokenizer,
    class_weights=class_weights,
)

print("\n Training complete!")
print(f"Training loss: {train_result.training_loss:.4f}")

# ============================================================================
# Save trained model
# ============================================================================

print("SAVING TRAINED MODEL")

save_model(model, tokenizer, MODEL_SAVE_DIR)

# ============================================================================
# Get test predictions and evaluate
# ============================================================================

print("EVALUATING ON TEST SET")

print("\nGetting test predictions...")
predictions, labels = get_test_predictions(trainer, test_dataset)

print(f"  Predictions shape: {predictions.shape}")
print(f"  Labels shape: {labels.shape}")

print_test_results(predictions, labels)

# ============================================================================
# Create visualizations
# ============================================================================

print("CREATING VISUALIZATIONS")

# Load original dataset for visualization (small dataset, OK to load)
dataset = load_plod_dataset()

print("\nPlotting label distribution...")
plot_label_distribution(dataset)

print("Plotting sequence length distribution...")
plot_sequence_length_distribution(dataset)

# ============================================================================
# Summary
# ============================================================================

print("TRAINING COMPLETE!")
print(f"\nModel saved to: {MODEL_SAVE_DIR}")
print(f"Plots saved to: {PLOTS_DIR}")
print(f"Logs saved to: {LOG_DIR}")
