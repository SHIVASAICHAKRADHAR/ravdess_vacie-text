# model_utils.py
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_audio_resources():
    """Load audio model and encoders"""
    audio_model = load_model('full_audio_metadata_model.h5')
    
    encoders = {}
    for enc_name in ['actor', 'emotion', 'intensity', 'modality', 
                    'repetition', 'statement', 'vocal']:
        with open(f'{enc_name}_encoder.pkl', 'rb') as f:
            encoders[enc_name] = pickle.load(f)
    
    return (audio_model, encoders['actor'], encoders['emotion'], 
            encoders['intensity'], encoders['modality'], 
            encoders['repetition'], encoders['statement'], 
            encoders['vocal'])

def load_text_resources():
    """Load text model resources"""
    text_model = load_model('text_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('label_encoder_text.pkl', 'rb') as f:
        label_enc = pickle.load(f)
    return text_model, tokenizer, label_enc

def parse_ravdess_filename(filename):
    """Parse RAVDESS filename into components"""
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    parts = name.split('-')
    if len(parts) != 7:
        raise ValueError(f"Filename '{filename}' doesn't conform to RAVDESS format")
    return {
        "modality": parts[0],
        "vocal": parts[1],
        "emotion": parts[2],
        "intensity": parts[3],
        "statement": parts[4],
        "repetition": parts[5],
        "actor": parts[6]
    }
# app.py
import streamlit as st
import model_utils as mu
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Set up the app
st.set_page_config(page_title="Multimodal Emotion Detector", layout="wide")
st.title("🎤📝 Multimodal Emotion Detection")

# Constants
RAVDESS_EMOTION_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

def plot_probabilities(probs, labels, title):
    """Create probability bar plot"""
    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, probs, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title)
    st.pyplot(fig)

# Create layout
col1, col2 = st.columns(2)

with col1:
    st.header("🎤 Audio Input")
    audio_file = st.file_uploader("Upload RAVDESS Audio (.wav)", type=["wav"])

with col2:
    st.header("📝 Text Input")
    text_input = st.text_area("Enter text here", height=150)

# Load models (moved after UI elements to show progress)
try:
    (audio_model, actor_enc, emotion_enc, intensity_enc, 
     modality_enc, repetition_enc, statement_enc, vocal_enc) = mu.load_audio_resources()
    text_model, tokenizer, label_enc_text = mu.load_text_resources()
    models_loaded = True
except Exception as e:
    st.error(f"Failed to load models: {str(e)}")
    models_loaded = False

if models_loaded:
    # Process inputs
    if audio_file:
        try:
            meta = mu.parse_ravdess_filename(audio_file.name)
            features = [
                modality_enc.transform([meta["modality"]])[0],
                vocal_enc.transform([meta["vocal"]])[0],
                emotion_enc.transform([meta["emotion"]])[0],
                intensity_enc.transform([meta["intensity"]])[0],
                statement_enc.transform([meta["statement"]])[0],
                repetition_enc.transform([meta["repetition"]])[0],
                actor_enc.transform([meta["actor"]])[0]
            ]
            proba = audio_model.predict(np.array([features]))[0]
            emotion_code = emotion_enc.inverse_transform([np.argmax(proba)])[0]
            audio_emotion = RAVDESS_EMOTION_MAP.get(emotion_code, emotion_code)
            
            st.success(f"Audio Emotion: {audio_emotion.capitalize()}")
            plot_probabilities(proba, list(RAVDESS_EMOTION_MAP.values()), "Audio Probabilities")
        except Exception as e:
            st.error(f"Audio processing error: {str(e)}")

    if text_input:
        try:
            seq = tokenizer.texts_to_sequences([text_input])
            padded = pad_sequences(seq, maxlen=100, padding='post')
            proba = text_model.predict(padded)[0]
            text_emotion = label_enc_text.inverse_transform([np.argmax(proba)])[0]
            
            st.success(f"Text Emotion: {text_emotion.capitalize()}")
            plot_probabilities(proba, label_enc_text.classes_, "Text Probabilities")
        except Exception as e:
            st.error(f"Text processing error: {str(e)}")
