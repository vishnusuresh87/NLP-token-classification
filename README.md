# NLP-token-classification
NLP token classification on biomedical document

Fine-Tuning PubMedBERT for Biomedical Named Entity Recognition on PLOD-CW-25

Biomedical literature contains dense and specialized terminology that is difficult to process at scale using manual curation. Automatic extraction of entities such as activities (e.g., interventions, procedures) and life forms (e.g., organisms, species) is critical for downstream tasks including knowledge graph construction, evidence synthesis, and clinical decision support.
The specific problem addressed in this experiment is:
Given a corpus of biomedical and clinical text, train a domain specific NER model that can reliably identify and classify tokens into four label types: O (outside), B-AC (beginning of Activity), B-LF (beginning of Life Form), and I-LF (inside Life Form).
Key objectives:
	Achieve high precision and recall across all entity labels, with particular emphasis on minority classes (B-LF, B-AC).
	Handle significant label imbalance (majority O class).
	Build a modular, reproducible pipeline that can be adapted to other NER tasks and datasets.
	Provide transparent evaluation and error analysis, including visualizations and sample predictions.
	Expose the trained model through an interactive inference interface (Streamlit app).



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
 PubMedNER Experiment documentation    - Full documentation
 README.md                  - 

 See EXECUTION_GUIDE.md for step-by-step execution guide
 See PubMedNER Experiment documentation for full documentation


 


