"""
visualization_utils.py 
Plotting and statistical analysis utilities
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from config import *

def print_dataset_stats(stats):
    """Print dataset statistics in formatted table"""
    print("DATASET STATISTICS")
    
    for split_name, split_stats in stats.items():
        print(f"\n{split_name.upper()} SET:")
        print(f"  Samples: {split_stats['num_samples']}")
        print(f"  Total tokens: {split_stats['total_tokens']}")
        print(f"  Avg tokens/sample: {split_stats['avg_tokens_per_sample']:.1f}")
        print(f"  Max sequence length: {split_stats['max_sequence_length']}")
        
        print(f"\n  Label distribution:")
        label_dist = split_stats['label_distribution']
        total = sum(label_dist.values())
        
        if total == 0:
            print("    (No labels found in this split)")
        else:
            for label_name in sorted(label_dist.keys()):
                count = label_dist[label_name]
                if count > 0:
                    pct = (count / total) * 100
                    print(f"    {label_name}: {count} ({pct:.2f}%)")


def print_class_weights(class_weights):
    """Print class weights for loss function"""
    print("CLASS WEIGHTS (for loss function)")
    print()
    
    for label_name in sorted(class_weights.keys()):
        weight = class_weights[label_name]
        print(f"  {label_name}: {weight:.4f}")


def print_test_results(predictions, labels, label_names=None):
    """
    Print test results with classification report.
    FIXED: Convert logits to predictions first.
    """
    if label_names is None:
        label_names = LABEL_NAMES
    
    print("TEST RESULTS")
    
    # Convert logits to predictions (if not already done)
    if len(predictions.shape) == 3:
        # Shape: (batch, seq_len, num_labels) -> (batch, seq_len)
        predictions = np.argmax(predictions, axis=2)
    
    # Convert predictions to label names
    true_predictions = []
    true_labels = []
    
    for prediction, label in zip(predictions, labels):
        for pred_id, label_id in zip(prediction, label):
            if label_id != -100:
                # Convert numpy types to Python int
                pred_id = int(pred_id)
                label_id = int(label_id)
                
                if pred_id < len(label_names):
                    true_predictions.append(label_names[pred_id])
                if label_id < len(label_names):
                    true_labels.append(label_names[label_id])
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, true_predictions, labels=label_names))


def plot_label_distribution(dataset):
    """Plot label distribution across splits - FIXED"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    splits = ['train', 'validation', 'test']
    
    for idx, split in enumerate(splits):
        if split not in dataset:
            axes[idx].text(0.5, 0.5, f'No {split} data', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            continue
        
        # Collect all labels for this split
        label_counts = {label: 0 for label in LABEL_NAMES}
        
        for example in dataset[split]:
            if 'ner_tags' in example:
                for label in example['ner_tags']:
                    if label in label_counts:
                        label_counts[label] += 1
        
        # Prepare data for plotting
        labels = list(label_counts.keys())
        counts = [label_counts[label] for label in labels]
        
        # Plot
        ax = axes[idx]
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        bars = ax.bar(labels, counts, color=colors[:len(labels)], edgecolor='black', alpha=0.8)
        
        # Add title and labels
        ax.set_title(f'{split.capitalize()} Set\n(n={len(dataset[split])})', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=11)
        ax.set_xlabel('Label', fontsize=11)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    import os
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    plt.savefig(f'{PLOTS_DIR}/label_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR}/label_distribution.png")


def plot_sequence_length_distribution(dataset):
    """Plot sequence length distribution"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    splits = ['train', 'validation', 'test']
    
    for idx, split in enumerate(splits):
        if split not in dataset:
            axes[idx].text(0.5, 0.5, f'No {split} data', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            continue
        
        lengths = []
        for example in dataset[split]:
            if 'tokens' in example:
                lengths.append(len(example['tokens']))
        
        if not lengths:
            continue
        
        ax = axes[idx]
        ax.hist(lengths, bins=30, color='#2E86AB', edgecolor='black', alpha=0.7)
        ax.set_title(f'{split.capitalize()} Set', fontsize=12, fontweight='bold')
        ax.set_xlabel('Sequence Length (tokens)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.axvline(np.mean(lengths), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(lengths):.1f}')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    import os
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    plt.savefig(f'{PLOTS_DIR}/sequence_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {PLOTS_DIR}/sequence_length_distribution.png")


def plot_confusion_matrix(predictions, label_ids):
    """Plot confusion matrix from predictions"""
    pred_tags = []
    true_tags = []
    
    for pred_seq, label_seq in zip(predictions, label_ids):
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id != -100 and label_id >= 0 and pred_id >= 0:
                if pred_id < len(LABEL_NAMES) and label_id < len(LABEL_NAMES):
                    true_tags.append(LABEL_NAMES[label_id])
                    pred_tags.append(LABEL_NAMES[pred_id])
    
    if not true_tags:
        print("Warning: No valid predictions to create confusion matrix")
        return
    
    cm = confusion_matrix(true_tags, pred_tags, labels=LABEL_NAMES)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax, 
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    import os
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    plt.savefig(f'{PLOTS_DIR}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR}/confusion_matrix.png")


def plot_training_curves(trainer):
    """Plot training loss and validation F1 curves"""
    if not hasattr(trainer, 'state') or not trainer.state.log_history:
        print("No training history available")
        return
    
    logs = trainer.state.log_history
    
    epochs = []
    losses = []
    f1_scores = []
    
    for log in logs:
        if 'loss' in log:
            losses.append(log['loss'])
        if 'eval_f1' in log:
            f1_scores.append(log['eval_f1'])
            if 'epoch' in log:
                epochs.append(log['epoch'])
    
    if not losses and not f1_scores:
        print("No training curves to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if losses:
        ax1.plot(losses, marker='o', linestyle='-', color='#2E86AB', linewidth=2, markersize=6)
        ax1.set_xlabel('Step', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    if f1_scores:
        if not epochs:
            epochs = list(range(1, len(f1_scores) + 1))
        ax2.plot(epochs, f1_scores, marker='s', linestyle='-', color='#A23B72', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('F1 Score', fontsize=11)
        ax2.set_title('Validation F1 Score', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    
    import os
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    plt.savefig(f'{PLOTS_DIR}/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS_DIR}/training_curves.png")


def extract_entities(words, labels):
    """Extract entities from predictions"""
    entities = []
    current_entity = None
    
    for word, label in zip(words, labels):
        if label == 'O':
            if current_entity:
                entities.append(current_entity)
                current_entity = None
        elif label.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            entity_type = label.split('-')[1]
            current_entity = {'type': entity_type, 'tokens': [word]}
        elif label.startswith('I-'):
            if current_entity:
                current_entity['tokens'].append(word)
    
    if current_entity:
        entities.append(current_entity)
    
    return entities
