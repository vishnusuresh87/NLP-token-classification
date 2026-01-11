"""
streamlit_app.py
Streamlit deployment with inference, logging, and monitoring
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import torch
from inference_utils import (
    load_trained_model_and_tokenizer,
    predict_word_level,
    extract_abbreviations_and_longforms,
    log_prediction,
    get_prediction_statistics,
    load_prediction_logs
)
from config import *

st.set_page_config(
    page_title="Biomedical Abbreviation Detector",
    page_icon="",
    layout="wide"
)

st.markdown("""
    <style>
        .main { max-width: 1200px; margin: 0 auto; }
        .entity-o { background-color: #e8e8e8; padding: 2px 6px; margin: 2px; border-radius: 4px; }
        .entity-ac { background-color: #ff6b6b; color: white; padding: 2px 6px; margin: 2px; border-radius: 4px; }
        .entity-lf { background-color: #51cf66; color: white; padding: 2px 6px; margin: 2px; border-radius: 4px; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_cached():
    return load_trained_model_and_tokenizer()

st.title("Biomedical Abbreviation & Long-Form Detection")
st.markdown("Powered by PubMedBERT - Token Classification for Biomedical Text")
st.markdown("---")

with st.sidebar:
    st.header("About")
    st.write("""
    This tool detects abbreviations and their long-form expansions in biomedical text.
    
    **Model:** PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)
    **Task:** Token classification (BIO tagging)
    **Labels:**
    - **B-AC**: Abbreviation/Acronym
    - **B-LF**: Long Form (start)
    - **I-LF**: Long Form (continuation)
    - **O**: Other
    """)
    
    st.header("Statistics")
    stats = get_prediction_statistics()
    if stats:
        st.metric("Total Predictions", stats['total_predictions'])
        st.metric("Total Words Processed", stats['total_words'])
        st.metric("Avg Words/Prediction", f"{stats['avg_words_per_prediction']:.1f}")

model, tokenizer = load_model_cached()

tab1, tab2, tab3 = st.tabs(["Prediction", "Examples", "Logs"])

with tab1:
    st.header("Enter Biomedical Text")
    
    text = st.text_area(
        "Input text:",
        placeholder="e.g., EPI = Echo planar imaging is used in MRI scans.",
        height=120,
        key="input_text"
    )
    
    if st.button("Detect Abbreviations", type="primary", use_container_width=True):
        if text.strip():
            try:
                words, labels, confidences = predict_word_level(text, model, tokenizer)
                
                log_prediction(text, words, labels, confidences)
                
                st.success("✓ Prediction complete!")
                
                st.subheader("Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Words Processed", len(words))
                with col2:
                    abbr_count = sum(1 for l in labels if l == 'B-AC')
                    st.metric("Abbreviations Found", abbr_count)
                with col3:
                    lf_count = sum(1 for l in labels if l in ['B-LF', 'I-LF'])
                    st.metric("Long Form Tokens", lf_count)
                
                st.subheader(" Highlighted Text")
                html_output = "<div style='line-height: 2.5;'>"
                for word, label, conf in zip(words, labels, confidences):
                    if label == 'O':
                        html_output += f'<span class="entity-o">{word}</span> '
                    elif label == 'B-AC':
                        html_output += f'<span class="entity-ac"><strong>{word}</strong> ({conf:.2f})</span> '
                    elif label in ['B-LF', 'I-LF']:
                        html_output += f'<span class="entity-lf"><strong>{word}</strong> ({conf:.2f})</span> '
                html_output += "</div>"
                st.markdown(html_output, unsafe_allow_html=True)
                
                st.subheader("Detailed Results Table")
                results_df = pd.DataFrame({
                    'Word': words,
                    'Label': labels,
                    'Confidence': [f"{c:.4f}" for c in confidences],
                    'Entity Type': [
                        'Abbreviation' if l == 'B-AC'
                        else 'Long Form' if l in ['B-LF', 'I-LF']
                        else 'Other'
                        for l in labels
                    ]
                })
                st.dataframe(results_df, use_container_width=True)
                
                abbr, lf, pairs = extract_abbreviations_and_longforms(words, labels)
                
                st.subheader("Detected Pairs")
                if pairs:
                    pairs_df = pd.DataFrame(pairs, columns=['Abbreviation', 'Long Form'])
                    st.dataframe(pairs_df, use_container_width=True)
                else:
                    st.info("No abbreviation-longform pairs detected in this text.")
                
                st.subheader("Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Abbreviations:**")
                    if abbr:
                        st.write(", ".join(abbr))
                    else:
                        st.write("None detected")
                with col2:
                    st.write("**Long Forms:**")
                    if lf:
                        st.write(", ".join(lf))
                    else:
                        st.write("None detected")
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        else:
            st.warning("Please enter some text to analyze.")

with tab2:
    st.header("Example Biomedical Texts")
    
    examples = [
        "EPI = Echo planar imaging is a fast MRI technique.",
        "The MRI uses RF = Radio Frequency pulses for excitation.",
        "BOLD = Blood-Oxygen-Level-Dependent imaging shows brain activity.",
        "PubMed contains MEDLINE citations and life science literature.",
        "NER = Named Entity Recognition is used for entity extraction.",
    ]
    
    selected_example = st.selectbox("Choose an example:", examples)
    
    if st.button("Run on Example", type="secondary", use_container_width=True):
        words, labels, confidences = predict_word_level(selected_example, model, tokenizer)
        log_prediction(selected_example, words, labels, confidences)
        
        results_df = pd.DataFrame({
            'Word': words,
            'Label': labels,
            'Confidence': [f"{c:.4f}" for c in confidences]
        })
        st.dataframe(results_df, use_container_width=True)

with tab3:
    st.header(" Prediction Logs")
    
    logs = load_prediction_logs()
    
    if logs:
        st.metric("Total Logged Predictions", len(logs))
        
        if st.button("Refresh Logs", use_container_width=True):
            st.rerun()
        
        for i, log in enumerate(reversed(logs[-10:])):
            with st.expander(f"Prediction {len(logs)-i} - {log['timestamp']}"):
                st.write(f"**Input:** {log['input_text']}")
                log_df = pd.DataFrame({
                    'Word': log['words'],
                    'Prediction': log['predictions'],
                    'Confidence': [f"{c:.4f}" for c in log['confidences']]
                })
                st.dataframe(log_df, use_container_width=True)
    else:
        st.info("No predictions logged yet. Make some predictions to see them here.")

st.markdown("---")
st.markdown("""
    <small>
    Natural Language Processing<br>
    Abbreviation and Long-Form Detection Experiment
    </small>
""", unsafe_allow_html=True)
