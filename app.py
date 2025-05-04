import streamlit as st
import model_utils as mu
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set up the Streamlit app configuration
st.set_page_config(page_title="Multimodal Emotion Detector", layout="wide")
st.title("🎤📝 Multimodal Emotion Detection")
st.markdown("This app predicts emotional expression from **RAVDESS** audio metadata and free-form text input.")

# Create two columns for inputs: audio and text
col1, col2 = st.columns(2)

with col1:
    st.header("🎤 Audio Input")
    audio_file = st.file_uploader("Upload RAVDESS Audio (.wav)", type=["wav"])

with col2:
    st.header("📝 Text Input")
    text_input = st.text_area("Enter text here", height=150)

# Divider for predictions section
st.markdown("---")
st.header("🔍 Predictions")

# Initialize placeholders for predicted emotions
audio_emotion = None
text_emotion = None

# Create containers for prediction outputs
audio_container = st.container()
text_container = st.container()

# Audio prediction
with audio_container:
    st.subheader("🎧 Audio Emotion Prediction")
    if audio_file is not None:
        try:
            audio_emotion = mu.predict_audio_emotion(audio_file)
            st.success(f"**Predicted Emotion (Audio):** {audio_emotion.capitalize()}", icon="🎙️")
        except Exception as e:
            st.error(f"Error processing audio file: {e}")
    else:
        st.info("No audio uploaded. Upload a RAVDESS file to get a prediction.")

# Text prediction
with text_container:
    st.subheader("✍️ Text Emotion Prediction")
    if text_input:
        try:
            text_emotion = mu.predict_text_emotion(text_input)
            st.success(f"**Predicted Emotion (Text):** {text_emotion.capitalize()}", icon="💬")
        except Exception as e:
            st.error(f"Error processing text input: {e}")
    else:
        st.info("No text entered. Enter text to get a prediction.")

# Combined prediction section
st.markdown("---")
combined_container = st.container()

with combined_container:
    st.subheader("🤝 Combined Prediction (Audio + Text)")
    if audio_file is not None and text_input:
        try:
            # Load models and encoders (cached)
            (audio_model,
             actor_enc,
             emotion_enc,
             intensity_enc,
             modality_enc,
             repetition_enc,
             statement_enc,
             vocal_enc) = mu.load_audio_resources()
            text_model, tokenizer, label_enc_text = mu.load_text_resources()

            # Prepare audio metadata features from filename
            meta = mu.parse_ravdess_filename(audio_file.name)
            modality_val = modality_enc.transform([meta["modality"]])[0]
            vocal_val = vocal_enc.transform([meta["vocal"]])[0]
            emotion_val = emotion_enc.transform([meta["emotion"]])[0]
            intensity_val = intensity_enc.transform([meta["intensity"]])[0]
            statement_val = statement_enc.transform([meta["statement"]])[0]
            repetition_val = repetition_enc.transform([meta["repetition"]])[0]
            actor_val = actor_enc.transform([meta["actor"]])[0]
            X_audio = np.array([[modality_val, vocal_val, emotion_val,
                                  intensity_val, statement_val, repetition_val, actor_val]])

            # Predict probability distribution for audio
            proba_audio = audio_model.predict(X_audio)[0]

            # Prepare text input features
            seq = tokenizer.texts_to_sequences([text_input])
            try:
                maxlen = text_model.input_shape[1]
            except Exception:
                maxlen = 100
            padded_seq = pad_sequences(seq, maxlen=maxlen, padding='post')
            proba_text = text_model.predict(padded_seq)[0]

            # Combine probabilities by averaging
            combined_proba = (proba_audio + proba_text) / 2
            combined_idx = np.argmax(combined_proba)
            combined_emotion = label_enc_text.inverse_transform([combined_idx])[0]

            st.success(f"**Combined Predicted Emotion:** {combined_emotion.capitalize()}", icon="🤖")
        except Exception as e:
            st.error(f"Error in combined prediction: {e}")
    else:
        if audio_file is None and not text_input:
            st.warning("Please provide audio and/or text input to get predictions.")
        elif audio_file is None:
            st.info("Provide an audio file to see combined results.")
        elif not text_input:
            st.info("Provide text input to see combined results.")




