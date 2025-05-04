# -*- coding: utf-8 -*-
import streamlit as st
import os
import tempfile
import traceback

from model_utils import (
    load_models,
    load_all_encoders,
    predict_multimodal
)

# Load models and encoders once
@st.cache_resource
def load_all():
    audio_model, text_model = load_models()
    encoders, text_encoders = load_all_encoders()
    return audio_model, text_model, encoders, text_encoders

audio_model, text_model, encoders, text_encoders = load_all()

# Page config
st.set_page_config(page_title="Multi-Modal Emotion Detector", layout="centered")
st.title("🎭 Multi-Modal Emotion Detector")
st.markdown("Upload a **RAVDESS audio file** (metadata-based) and/or enter **text** to predict emotion.")

# Input section
col1, col2 = st.columns(2)

with col1:
    uploaded_audio = st.file_uploader("Upload RAVDESS `.wav` file (optional)", type=["wav"])
    audio_path = None
    if uploaded_audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_audio.getbuffer())
            audio_path = tmp_file.name

with col2:
    user_text = st.text_area("Enter a sentence for emotion detection (optional)", height=150)

# Predict button
if st.button("🔍 Predict"):
    if not uploaded_audio and not user_text.strip():
        st.warning("Please upload an audio file or enter text.")
    else:
        try:
            results = predict_multimodal(
                audio_path if uploaded_audio else None,
                user_text if user_text.strip() else "",
                audio_model,
                text_model,
                encoders,
                text_encoders
            )

            st.subheader("🎯 Predictions")
            if uploaded_audio:
                st.success(f"**Audio Metadata Emotion**: {results['audio_emotion']}")
            if user_text.strip():
                st.success(f"**Text Emotion**: {results['text_emotion']}")
            if uploaded_audio and user_text.strip():
                st.info(f"**Combined Final Emotion**: {results['final_emotion']}")

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.text(traceback.format_exc())

