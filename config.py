"""
config.py
Configuration file for PLOD-CW-25 Abbreviation Detection Experiment
"""

# Dataset Configuration
DATASET_NAME = 'surrey-nlp/PLOD-CW-25'
TRAIN_SPLIT = 'train'
VALIDATION_SPLIT = 'validation'
TEST_SPLIT = 'test'

# Model Configuration
MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
NUM_LABELS = 4
#LABEL_NAMES = {0: 'O', 1: 'B-AC', 2: 'B-LF', 3: 'I-LF'}
LABEL_NAMES = ['O', 'B-AC', 'B-LF', 'I-LF']
#LABEL_TO_ID = {v: k for k, v in LABEL_NAMES.items()}
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_NAMES)}
ID_TO_LABEL = {idx: label for idx, label in enumerate(LABEL_NAMES)}


# Tokenizer Configuration
MAX_LENGTH = 512
PADDING = True
TRUNCATION = True

# Training Configuration
NUM_EPOCHS = 5            
LEARNING_RATE = 5e-5         
BATCH_SIZE_TRAIN = 32        
BATCH_SIZE_EVAL = 32       

#LEARNING_RATE = 2e-5
#BATCH_SIZE_TRAIN = 16
#BATCH_SIZE_EVAL = 32
#NUM_EPOCHS = 3

WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
EARLY_STOPPING_PATIENCE = 2
FP16 = True
DATALOADER_NUM_WORKERS = 4
SEED = 42

# Paths
OUTPUT_DIR = './plod_results'
MODEL_SAVE_DIR = './best_pubmedbert_plod'
LOG_DIR = './logs'
PLOTS_DIR = './plots'
PREDICTIONS_LOG = './prediction_logs.json'

# Evaluation
EVALUATION_STRATEGY = 'epoch'
SAVE_STRATEGY = 'epoch'
SAVE_TOTAL_LIMIT = 2
METRIC_FOR_BEST_MODEL = 'f1'
LOAD_BEST_MODEL_AT_END = True

# Device
DEVICE = 'cpu'  # Change to 'cpu' or 'cuda' based on availability
