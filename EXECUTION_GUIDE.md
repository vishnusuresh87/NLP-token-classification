"""
EXECUTION_GUIDE.md
Step-by-step execution guide
"""



## File Summary

 **9 modular Python files**:

```
 config.py                  - Configuration (hyperparameters)
 data_utils.py              - Data utilities
 model_utils.py             - Model training utilities
 visualization_utils.py     - Visualization utilities
 inference_utils.py         - Inference utilities
 dataset_analysis.py      - executable
 data_preprocessing.py    - executable
 model_training.py        - executable
 evaluation_and_analysis.py - executable
 streamlit_app.py         - (Streamlit deployment)
 requirements.txt           - Dependencies
 README.md                  - Full documentation
```

---

## How to Execute

### Setup

```python
# Install dependencies
pip install -q torch transformers datasets seqeval scikit-learn pandas numpy matplotlib seaborn streamlit

# Create working directory
import os
os.makedirs('./plod_results', exist_ok=True)
os.makedirs('./plots', exist_ok=True)
os.makedirs('./best_pubmedbert_plod', exist_ok=True)
```

### Dataset Analysis

```python
%run dataset_analysis.py
```


### Data Preprocessing

```python
%run data_preprocessing.py
```


### Model Training

```python
%run model_training.py
```


**NOTE:** This step takes 1-3 hours on GPU.

### Evaluation & Error Analysis

```python
%run evaluation_and_analysis.py
```


### If Training is Too Slow
Edit `config.py`:
```python
# Reduce batch size
BATCH_SIZE_TRAIN = 8  

# Or reduce epochs
NUM_EPOCHS = 2 
```

### If Out of Memory
```python
# In config.py
MAX_LENGTH = 256  
BATCH_SIZE_TRAIN = 8 
FP16 = True 
```

---

## PStreamlit Deployment (Local or Hugging Face Spaces)

### Option A: Local Execution
```bash
streamlit run 5_streamlit_app.py
```

Visit `http://localhost:8501` in browser.

### Option B: Hugging Face Spaces

1. Upload trained model to Hugging Face:
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./best_pubmedbert_plod",
    repo_id="your-username/plod-pubmedbert",
    repo_type="model"
)
```

2. Update `config.py`:
```python
MODEL_SAVE_DIR = "your-username/plod-pubmedbert"
```

3. Create GitHub repo with all files + `streamlit_app.py`

4. Deploy to Spaces:
   - Go to huggingface.co/spaces
   - Create new Space
   - Select Streamlit runtime
   - Connect to GitHub repo

---



### View Training Curves

```python
from PIL import Image
img = Image.open('./plots/training_curves.png')
img.show()
```

---

## Output File Locations

```
After dataset_analysis.py:
- ./plots/label_distribution.png
- ./plots/sequence_length_distribution.png

After data_preprocessing.py:
- tokenized_datasets.pkl
- class_weights.pkl

After model_training.py:
- ./best_pubmedbert_plod/  (trained model)
- ./plots/training_curves.png

After evaluation_and_analysis.py:
- ./plots/confusion_matrix.png
- Console output (error analysis)

After streamlit_app.py:
- prediction_logs.json (user predictions)
```

---

## Downloading Results 

After all executions been complete, download these files:

```python
# Download plots for report
import shutil
shutil.make_archive('plots', 'zip', './plots')

# Download trained model
shutil.make_archive('model', 'zip', './best_pubmedbert_plod')

# Download logs
import shutil
shutil.copy('prediction_logs.json', 'prediction_logs.json')
```


## Troubleshooting

### Issue: "CUDA out of memory"
**Solution:**
```python
# In config.py
BATCH_SIZE_TRAIN = 8
BATCH_SIZE_EVAL = 16
MAX_LENGTH = 256
```

### Issue: "ModuleNotFoundError: No module named 'config'"
**Solution:** Make sure all `.py` files are in the same directory/notebook session.

### Issue: "Training too slow"
**Solution:** Enable mixed precision (already enabled in config: `FP16 = True`)

### Issue: "F1 score is low (< 0.75)"
**Solution:** Check:
1. Label alignment is correct (print tokenized_datasets sample)
2. Class weights are being used (check model_utils.py)
3. Training loss is decreasing (check training_curves.png)

---




