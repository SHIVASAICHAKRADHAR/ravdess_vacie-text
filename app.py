import streamlit as st
import numpy as np
import librosa
from model_utils import load_audio_resources, load_text_resources
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load models and encoders
audio_model, audio_encoders = load_audio_resources()
text_model, tokenizer, label_encoder_text = load_text_resources()

st.title("🎵 Audio & Text Emotion Recognition")

# Upload .wav file
audio_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

# Optional text input
user_text = st.text_input("Enter a sentence (optional for text-based prediction):")

# Button
if st.button("🔍 Predict"):

    if audio_file:
        try:
            # Load audio
            y, sr = librosa.load(audio_file, sr=22050)

            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfcc_processed = np.mean(mfcc.T, axis=0)
            mfcc_processed = mfcc_processed.reshape(1, -1)

            # Predict with audio model
            audio_pred = audio_model.predict(mfcc_processed)
            emotion_idx = np.argmax(audio_pred)
            emotion_label = audio_encoders['emotion'].inverse_transform([emotion_idx])[0]

            st.success(f"🎧 Predicted Emotion from Audio: **{emotion_label}**")

        except Exception as e:
            st.error(f"Error processing audio file: {e}")

    # Optional text model prediction
    if user_text and tokenizer and text_model:
        try:
            seq = tokenizer.texts_to_sequences([user_text])
            padded = pad_sequences(seq, maxlen=100, padding='post')
            text_pred = text_model.predict(padded)
            pred_label = label_encoder_text.inverse_transform([np.argmax(text_pred)])
            st.success(f"📚 Predicted Emotion from Text: **{pred_label[0]}**")
        except Exception as e:
            st.error(f"Error processing text: {e}")
