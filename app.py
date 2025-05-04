import streamlit as st
import os
from model_utils import load_all_encoders, load_models, predict_multimodal
import numpy as np

# Set up file upload configuration
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

# Load models and encoders
encoders, text_encoders = load_all_encoders()
audio_model, text_model = load_models()

# Utility function to check allowed file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Streamlit UI components
st.title("Multimodal Emotion Prediction")

# Audio file upload
audio_file = st.file_uploader("Upload an Audio File", type=['wav', 'mp3'])

# Text input
text_input = st.text_area("Enter Text")

# Run prediction when both audio and text are provided
if audio_file is not None and text_input:
    # Save uploaded audio file
    audio_filename = os.path.join(UPLOAD_FOLDER, audio_file.name)
    with open(audio_filename, "wb") as f:
        f.write(audio_file.getbuffer())

    # Get predictions
    predictions = predict_multimodal(audio_filename, text_input, audio_model, text_model, encoders, text_encoders)
    
    # Display the results
    st.subheader("Audio Emotion Prediction:")
    st.write(f"Predicted emotion from audio: {predictions['audio_emotion']}")
    
    st.subheader("Text Emotion Prediction:")
    st.write(f"Predicted emotion from text: {predictions['text_emotion']}")
    
    st.subheader("Combined Emotion Prediction:")
    st.write(f"Final emotion prediction: {predictions['final_emotion']}")
    
    # Clean up uploaded file
    os.remove(audio_filename)

