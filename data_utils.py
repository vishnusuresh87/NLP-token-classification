"""
data_utils.py 
Data loading and preprocessing for PLOD-CW dataset
"""

from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from config import *

def load_plod_dataset():
    """Load PLOD-CW-25 dataset from Hugging Face"""
    dataset = load_dataset(DATASET_NAME)
    return dataset


def analyze_dataset(dataset):
    """Analyze dataset and return statistics"""
    stats = {}
    
    for split_name in ['train', 'validation', 'test']:
        if split_name not in dataset:
            continue
            
        split = dataset[split_name]
        num_samples = len(split)
        total_tokens = 0
        label_counts = {label: 0 for label in LABEL_NAMES}
        max_length = 0
        
        for example in split:
            if 'tokens' not in example or 'ner_tags' not in example:
                continue
                
            tokens = example['tokens']
            labels = example['ner_tags']
            
            total_tokens += len(tokens)
            max_length = max(max_length, len(tokens))
            
            for label in labels:
                if label in label_counts:
                    label_counts[label] += 1
        
        # Keep labels as strings in stats dict
        label_distribution = label_counts
        
        avg_tokens = total_tokens / num_samples if num_samples > 0 else 0
        
        stats[split_name] = {
            'num_samples': num_samples,
            'total_tokens': total_tokens,
            'avg_tokens_per_sample': avg_tokens,
            'max_sequence_length': max_length,
            'label_distribution': label_distribution
        }
    
    return stats


def load_tokenizer():
    """Load PubMedBERT tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer


def tokenize_and_align_labels(examples, tokenizer):
    """Tokenize text and align labels with subword tokens"""
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_LENGTH,
        padding='max_length'
    )
    
    labels = []
    for i, label_seq in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens - ignore
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word - use the label
                label_str = label_seq[word_idx]
                label_id = LABEL_TO_ID.get(label_str, 0)
                label_ids.append(label_id)
            else:
                # Continuation subword - ignore
                label_ids.append(-100)
            
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs['labels'] = labels
    return tokenized_inputs


def preprocess_dataset(dataset, tokenizer):
    """Preprocess entire dataset"""
    tokenized_datasets = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        batch_size=32,
        remove_columns=['tokens', 'ner_tags']
    )
    
    return tokenized_datasets


def get_class_weights(dataset):
    """Compute class weights from training set"""
    if 'train' not in dataset:
        return {label: 1.0 for label in LABEL_NAMES}
    
    train_split = dataset['train']
    label_counts = {label: 0 for label in LABEL_NAMES}
    total_labels = 0
    
    for example in train_split:
        if 'ner_tags' not in example:
            continue
            
        for label in example['ner_tags']:
            if label in label_counts:
                label_counts[label] += 1
                total_labels += 1
    
    class_weights = {}
    for label in LABEL_NAMES:
        count = label_counts.get(label, 0)
        
        if count > 0 and total_labels > 0:
            weight = total_labels / (NUM_LABELS * count)
        else:
            weight = 1.0
        
        class_weights[label] = weight
    
    return class_weights


def compute_metrics_seqeval(predictions, labels):
    """Compute seqeval metrics for sequence labeling"""
    from seqeval.metrics import classification_report, f1_score
    
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = []
    true_labels = []
    
    for prediction, label in zip(predictions, labels):
        for pred_id, label_id in zip(prediction, label):
            if label_id != -100:
                if pred_id < len(LABEL_NAMES):
                    true_predictions.append(LABEL_NAMES[pred_id])
                if label_id < len(LABEL_NAMES):
                    true_labels.append(LABEL_NAMES[label_id])
    
    return {
        'f1': f1_score(true_labels, true_predictions),
        'report': classification_report(true_labels, true_predictions)
    }
