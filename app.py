import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from model_utils import load_audio_resources, load_text_resources, parse_ravdess_filename
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.audio_resources = None
    st.session_state.text_resources = None

# Page configuration
st.set_page_config(page_title="Multimodal Emotion Recognition", layout="wide")
st.title("🎤📝 Multimodal Emotion Recognition")

# Load models button
if st.button("🔄 Load Models"):
    with st.spinner("Loading models..."):
        try:
            audio_model, audio_encoders = load_audio_resources()
            text_model, tokenizer, label_encoder = load_text_resources()
            
            if audio_model and text_model:
                st.session_state.audio_resources = (audio_model, audio_encoders)
                st.session_state.text_resources = (text_model, tokenizer, label_encoder)
                st.session_state.models_loaded = True
                st.success("Models loaded successfully!")
            else:
                st.error("Failed to load one or more models")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")

# Main interface
if st.session_state.models_loaded:
    audio_model, audio_encoders = st.session_state.audio_resources
    text_model, tokenizer, label_encoder = st.session_state.text_resources

    col1, col2 = st.columns(2)

    with col1:
        st.header("🎤 Audio Analysis")
        audio_file = st.file_uploader("Upload RAVDESS audio file", type=["wav"])
        
        if audio_file:
            try:
                # Parse filename metadata
                meta = parse_ravdess_filename(audio_file.name)
                st.write("Audio metadata:", meta)
                
                # Extract features from audio
                y, sr = librosa.load(audio_file, sr=22050)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                mfcc_processed = np.mean(mfcc.T, axis=0).reshape(1, -1)
                
                # Predict
                audio_pred = audio_model.predict(mfcc_processed)
                emotion_idx = np.argmax(audio_pred)
                emotion_label = audio_encoders['emotion'].inverse_transform([emotion_idx])[0]
                
                st.success(f"Predicted emotion: {emotion_label}")
                
                # Plot probabilities
                fig, ax = plt.subplots()
                ax.bar(range(len(audio_pred[0])), audio_pred[0])
                ax.set_title("Emotion Probabilities")
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Audio processing error: {str(e)}")

    with col2:
        st.header("📝 Text Analysis")
        user_text = st.text_area("Enter text for emotion analysis")
        
        if user_text:
            try:
                # Process text
                seq = tokenizer.texts_to_sequences([user_text])
                padded = pad_sequences(seq, maxlen=100, padding='post')
                
                # Predict
                text_pred = text_model.predict(padded)
                pred_label = label_encoder.inverse_transform([np.argmax(text_pred)])[0]
                
                st.success(f"Predicted emotion: {pred_label}")
                
                # Plot probabilities
                fig, ax = plt.subplots()
                ax.bar(range(len(text_pred[0])), text_pred[0])
                ax.set_title("Emotion Probabilities")
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Text processing error: {str(e)}")

else:
    st.warning("Please load the models first using the button above")

# Requirements note
st.sidebar.markdown("### Requirements")
st.sidebar.code("""
pip install streamlit librosa tensorflow numpy matplotlib joblib
""")
