# app.py
import streamlit as st
import model_utils as mu
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import os

# Set up the Streamlit app configuration
st.set_page_config(page_title="Multimodal Emotion Detector", layout="wide")
st.title("🎤📝 Multimodal Emotion Detection")
st.markdown("This app predicts emotional expression from **RAVDESS** audio metadata and free-form text input.")

# Constants
OFFENSIVE_WORDS = ["fuck", "bitch", "asshole", "shit", "damn"]
RAVDESS_EMOTION_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

def contains_offensive_language(text):
    """Check for offensive words in text"""
    text_lower = text.lower()
    return any(word in text_lower for word in OFFENSIVE_WORDS)

def plot_probabilities(probs, labels, title):
    """Create a bar plot of emotion probabilities"""
    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, probs, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Probability')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    st.pyplot(fig)

# Create input columns
col1, col2 = st.columns(2)

with col1:
    st.header("🎤 Audio Input")
    audio_file = st.file_uploader("Upload RAVDESS Audio (.wav)", type=["wav"])
    
with col2:
    st.header("📝 Text Input")
    text_input = st.text_area("Enter text here", height=150)
    if text_input and contains_offensive_language(text_input):
        st.warning("⚠️ Offensive language may affect emotion detection accuracy")

# Initialize predictions
audio_emotion = text_emotion = None
audio_proba = text_proba = None

try:
    # Load models
    (audio_model, actor_enc, emotion_enc, intensity_enc, 
     modality_enc, repetition_enc, statement_enc, vocal_enc) = mu.load_audio_resources()
    text_model, tokenizer, label_enc_text = mu.load_text_resources()
    
    # Get label information
    text_labels = label_enc_text.classes_
    audio_labels = list(RAVDESS_EMOTION_MAP.values())
    
except Exception as e:
    st.error(f"Failed to load models: {str(e)}")
    st.stop()

# Audio processing
with col1:
    if audio_file:
        try:
            with st.spinner('Processing audio...'):
                # Get metadata from filename
                meta = mu.parse_ravdess_filename(audio_file.name)
                
                # Encode features
                features = [
                    modality_enc.transform([meta["modality"]])[0],
                    vocal_enc.transform([meta["vocal"]])[0],
                    emotion_enc.transform([meta["emotion"]])[0],
                    intensity_enc.transform([meta["intensity"]])[0],
                    statement_enc.transform([meta["statement"]])[0],
                    repetition_enc.transform([meta["repetition"]])[0],
                    actor_enc.transform([meta["actor"]])[0]
                ]
                
                # Predict
                X_audio = np.array([features])
                proba = audio_model.predict(X_audio)
                
                # Handle multi-output models
                if isinstance(proba, list):
                    proba = proba[0]  # Use first output head for emotion
                
                pred_idx = np.argmax(proba, axis=1)[0]
                pred_code = emotion_enc.inverse_transform([pred_idx])[0]
                audio_emotion = RAVDESS_EMOTION_MAP.get(pred_code, pred_code)
                audio_proba = proba[0]
                
                st.success(f"Predicted Emotion (Audio): **{audio_emotion.capitalize()}**")
                plot_probabilities(audio_proba, audio_labels, "Audio Emotion Probabilities")
                
        except Exception as e:
            st.error(f"Audio processing error: {str(e)}")

# Text processing
with col2:
    if text_input and not contains_offensive_language(text_input):
        try:
            with st.spinner('Analyzing text...'):
                # Tokenize and pad
                seq = tokenizer.texts_to_sequences([text_input])
                maxlen = text_model.input_shape[1] if hasattr(text_model.input_shape, '__len__') else 100
                padded = pad_sequences(seq, maxlen=maxlen, padding='post')
                
                # Predict
                text_proba = text_model.predict(padded)[0]
                pred_idx = np.argmax(text_proba)
                text_emotion = label_enc_text.inverse_transform([pred_idx])[0]
                
                st.success(f"Predicted Emotion (Text): **{text_emotion.capitalize()}**")
                plot_probabilities(text_proba, text_labels, "Text Emotion Probabilities")
                
        except Exception as e:
            st.error(f"Text processing error: {str(e)}")

# Combined prediction
if audio_file and text_input and not contains_offensive_language(text_input):
    try:
        st.subheader("🤝 Combined Prediction")
        
        # Align probabilities if needed
        if len(audio_proba) != len(text_proba):
            st.warning("Note: Using text emotion categories for combined prediction")
            combined_emotion = text_emotion
        else:
            combined_proba = (audio_proba + text_proba) / 2
            combined_idx = np.argmax(combined_proba)
            combined_emotion = text_labels[combined_idx]
            plot_probabilities(combined_proba, text_labels, "Combined Probabilities")
        
        st.success(f"Final Prediction: **{combined_emotion.capitalize()}**")
        
    except Exception as e:
        st.error(f"Combined prediction error: {str(e)}")

# Debug info
with st.expander("Debug Information"):
    if audio_file:
        st.write("Audio Metadata:", mu.parse_ravdess_filename(audio_file.name))
    if text_input:
        st.write("Text Processed:", text_input)
    st.write("Text Model Classes:", text_labels)
    st.write("Audio Model Classes:", audio_labels)

# Requirements note
st.markdown("---")
st.caption("Note: Requires matplotlib, tensorflow, numpy, and other dependencies")
