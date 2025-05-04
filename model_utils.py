# model_utils.py

import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

@st.cache_resource
def load_audio_resources():
    """
    Load the audio metadata model and label encoders for RAVDESS file metadata.
    Uses @st.cache_resource to load once for efficiency.
    """
    # Load pre-trained Keras model for audio metadata (based on filename features)
    audio_model = load_model('full_audio_metadata_model.h5')
    # Load LabelEncoders for each part of the metadata
    with open('actor_encoder.pkl', 'rb') as f:
        actor_enc = pickle.load(f)
    with open('emotion_encoder.pkl', 'rb') as f:
        emotion_enc = pickle.load(f)
    with open('intensity_encoder.pkl', 'rb') as f:
        intensity_enc = pickle.load(f)
    with open('modality_encoder.pkl', 'rb') as f:
        modality_enc = pickle.load(f)
    with open('repetition_encoder.pkl', 'rb') as f:
        repetition_enc = pickle.load(f)
    with open('statement_encoder.pkl', 'rb') as f:
        statement_enc = pickle.load(f)
    with open('vocal_encoder.pkl', 'rb') as f:
        vocal_enc = pickle.load(f)
    return audio_model, actor_enc, emotion_enc, intensity_enc, modality_enc, repetition_enc, statement_enc, vocal_enc

@st.cache_resource
def load_text_resources():
    """
    Load the text model, tokenizer, and label encoder for text emotion prediction.
    Uses @st.cache_resource to load once for efficiency.
    """
    # Load pre-trained text classification model
    text_model = load_model('text_model.h5')
    # Load tokenizer and label encoder for text processing
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('label_encoder_text.pkl', 'rb') as f:
        label_enc_text = pickle.load(f)
    return text_model, tokenizer, label_enc_text

def parse_ravdess_filename(filename):
    """
    Parse a RAVDESS filename into its 7 metadata components.
    Expected format (7 parts): 
    Modality-Vocal-Emotion-Intensity-Statement-Repetition-Actor (e.g., "03-01-06-01-02-01-12.wav").
    Returns a dict with keys: modality, vocal, emotion, intensity, statement, repetition, actor.
    """
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    parts = name.split('-')
    if len(parts) != 7:
        raise ValueError(f"Filename '{filename}' does not conform to RAVDESS naming convention.")
    keys = ["modality", "vocal", "emotion", "intensity", "statement", "repetition", "actor"]
    return dict(zip(keys, parts))

def predict_audio_emotion(file):
    """
    Given a RAVDESS audio file (UploadedFile or filepath string), extract metadata from filename
    and predict the emotion using the audio metadata model.
    Returns the predicted emotion label (string).
    """
    try:
        # Load models and encoders (cached)
        (audio_model,
         actor_enc,
         emotion_enc,
         intensity_enc,
         modality_enc,
         repetition_enc,
         statement_enc,
         vocal_enc) = load_audio_resources()

        # Determine the filename string
        if hasattr(file, 'name'):
            filename = file.name
        else:
            filename = str(file)
        # Parse metadata from the filename
        meta = parse_ravdess_filename(filename)

        # Encode each metadata component into numeric form
        modality_val = modality_enc.transform([meta["modality"]])[0]
        vocal_val = vocal_enc.transform([meta["vocal"]])[0]
        emotion_val = emotion_enc.transform([meta["emotion"]])[0]
        intensity_val = intensity_enc.transform([meta["intensity"]])[0]
        statement_val = statement_enc.transform([meta["statement"]])[0]
        repetition_val = repetition_enc.transform([meta["repetition"]])[0]
        actor_val = actor_enc.transform([meta["actor"]])[0]

        # Prepare the input array for prediction (shape 1x7)
        X = np.array([[modality_val, vocal_val, emotion_val,
                       intensity_val, statement_val, repetition_val, actor_val]])

        # Predict probability distribution over emotion classes
        proba = audio_model.predict(X)
        # Determine the class index with highest probability
        pred_idx = np.argmax(proba[2], axis=1)[0]  # Use the correct index based on your model's outputs

        # Decode the predicted class index to original label (RAVDESS emotion code)
        pred_code = emotion_enc.inverse_transform([pred_idx])[0]

        # Map RAVDESS emotion code to human-readable emotion name
        emotion_map = {
            "01": "neutral",
            "02": "calm",
            "03": "happy",
            "04": "sad",
            "05": "angry",
            "06": "fearful",
            "07": "disgust",
            "08": "surprised"
        }
        return emotion_map.get(pred_code, pred_code)
    except Exception as e:
        # On error (e.g., bad filename), propagate or handle as needed
        raise e

def predict_text_emotion(text):
    """
    Given input text, preprocess using the tokenizer and predict emotion using the text model.
    Returns the predicted emotion label (string).
    """
    try:
        # Load text model, tokenizer, and label encoder (cached)
        text_model, tokenizer, label_enc_text = load_text_resources()

        # Convert text to sequence of tokens
        seq = tokenizer.texts_to_sequences([text])
        # Determine the required sequence length from model input (if available)
        try:
            maxlen = text_model.input_shape[1]
        except Exception:
            maxlen = 100  # fallback if input shape is not fixed
        # Pad sequence to the required length
        padded = pad_sequences(seq, maxlen=maxlen, padding='post')

        # Predict probability distribution over emotion classes
        proba = text_model.predict(padded)
        # Find class index with highest probability
        pred_idx = np.argmax(proba, axis=1)[0]
        # Decode index to emotion label using label encoder
        pred_label = label_enc_text.inverse_transform([pred_idx])[0]
        return pred_label
    except Exception as e:
        raise e
