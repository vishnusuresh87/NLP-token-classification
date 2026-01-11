"""
inference_utils.py
Inference and deployment utilities
"""

import torch
import json
import datetime
from transformers import AutoTokenizer, AutoModelForTokenClassification
from config import *


def load_trained_model_and_tokenizer():
    """Load trained model and tokenizer from saved directory"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_SAVE_DIR)
    return model, tokenizer


def predict_word_level(text, model, tokenizer):
    """
    Predict word-level labels from input text.
    Handles subword tokenization and reconstructs word-level predictions.
    
    Args:
        text (str): Input text
        model: Trained PubMedBERT model
        tokenizer: PubMedBERT tokenizer
    
    Returns:
        words (list): Original words
        labels (list): Predicted labels at word level
        confidences (list): Prediction confidences
    """
    words = text.split()
    
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors='pt',
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**encoding)
    
    predictions = torch.argmax(outputs.logits, dim=2)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=2)
    
    subword_preds = predictions[0].cpu().numpy()
    subword_probs = probabilities[0].cpu().numpy()
    
    word_ids = encoding.word_ids()
    word_predictions = []
    word_confidences = []
    
    for word_idx in range(len(words)):
        for sub_idx, wid in enumerate(word_ids):
            if wid == word_idx:
                pred_label = LABEL_NAMES[int(subword_preds[sub_idx])]
                confidence = float(subword_probs[sub_idx][int(subword_preds[sub_idx])])
                word_predictions.append(pred_label)
                word_confidences.append(confidence)
                break
    
    return words, word_predictions, word_confidences


def extract_abbreviations_and_longforms(words, labels):
    """
    Extract abbreviations and long forms from predicted labels.
    
    Returns:
        abbreviations (list): List of detected abbreviations
        long_forms (list): List of detected long forms
        pairs (list): List of (abbreviation, long form) tuples if co-occurring
    """
    abbreviations = []
    long_forms = []
    pairs = []
    
    i = 0
    last_abbreviation = None
    
    while i < len(labels):
        if labels[i] == 'B-AC':
            last_abbreviation = words[i]
            abbreviations.append(words[i])
        elif labels[i] == 'B-LF':
            lf_tokens = [words[i]]
            i += 1
            while i < len(labels) and labels[i] == 'I-LF':
                lf_tokens.append(words[i])
                i += 1
            
            long_form = ' '.join(lf_tokens)
            long_forms.append(long_form)
            
            if last_abbreviation:
                pairs.append((last_abbreviation, long_form))
            
            i -= 1
        
        i += 1
    
    return abbreviations, long_forms, pairs


def log_prediction(text, words, labels, confidences):
    """
    Log user input and predictions to JSON file.
    Format: JSON Lines (one JSON object per line, newline-delimited)
    """
    os.makedirs(os.path.dirname(PREDICTIONS_LOG) or '.', exist_ok=True)
    
    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'input_text': text,
        'words': words,
        'predictions': labels,
        'confidences': confidences
    }
    
    with open(PREDICTIONS_LOG, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


def load_prediction_logs():
    """Load all prediction logs from JSON file"""
    logs = []
    
    if not os.path.exists(PREDICTIONS_LOG):
        return logs
    
    with open(PREDICTIONS_LOG, 'r') as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))
    
    return logs


def get_prediction_statistics():
    """Get statistics from prediction logs"""
    logs = load_prediction_logs()
    
    if not logs:
        return None
    
    total_predictions = len(logs)
    total_words = sum(len(log['words']) for log in logs)
    
    return {
        'total_predictions': total_predictions,
        'total_words': total_words,
        'avg_words_per_prediction': total_words / total_predictions if total_predictions > 0 else 0
    }


import os
