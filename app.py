
pip install matplotlib
import streamlit as st
import model_utils as mu
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import matplotlib.pyplot as plt
matplotlib>=3.0.0
streamlit
numpy
tensorflow
scikit-learn

# Set up the Streamlit app configuration
st.set_page_config(page_title="Multimodal Emotion Detector", layout="wide")
st.title("🎤📝 Multimodal Emotion Detection")
st.markdown("This app predicts emotional expression from **RAVDESS** audio metadata and free-form text input.")

# Content filter for offensive language
OFFENSIVE_WORDS = ["fuck", "bitch", "asshole", "shit", "damn"]  # Add more as needed

def contains_offensive_language(text):
    return any(word in text.lower() for word in OFFENSIVE_WORDS)

def plot_emotion_probabilities(probs, labels, title):
    fig, ax = plt.subplots()
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, probs, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # highest probability at top
    ax.set_xlabel('Probability')
    ax.set_title(title)
    st.pyplot(fig)

# Create two columns for inputs: audio and text
col1, col2 = st.columns(2)

with col1:
    st.header("🎤 Audio Input")
    audio_file = st.file_uploader("Upload RAVDESS Audio (.wav)", type=["wav"], key="audio_uploader")

with col2:
    st.header("📝 Text Input")
    text_input = st.text_area("Enter text here", height=150, key="text_input")
    if text_input and contains_offensive_language(text_input):
        st.warning("⚠️ Offensive language may affect emotion detection accuracy")

# Divider for predictions section
st.markdown("---")
st.header("Predictions")

# Initialize placeholders for predicted emotions
audio_emotion = None
text_emotion = None
audio_proba = None
text_proba = None

# Load models and encoders (moved here to load once)
try:
    (audio_model,
     actor_enc,
     emotion_enc,
     intensity_enc,
     modality_enc,
     repetition_enc,
     statement_enc,
     vocal_enc) = mu.load_audio_resources()
    text_model, tokenizer, label_enc_text = mu.load_text_resources()
    
    # Get label classes for both models
    audio_labels = emotion_enc.classes_
    text_labels = label_enc_text.classes_
    
except Exception as e:
    st.error(f"Failed to load models: {str(e)}")
    st.stop()

# Audio prediction column
with col1:
    if audio_file is not None:
        try:
            # Parse metadata and get prediction
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
            
            with st.spinner('Processing audio...'):
                audio_proba = audio_model.predict(X_audio)[0]
                audio_emotion = audio_labels[np.argmax(audio_proba)]
                
                st.success(f"Predicted Emotion (Audio): **{audio_emotion.capitalize()}**", icon="🎙️")
                plot_emotion_probabilities(audio_proba, audio_labels, "Audio Emotion Probabilities")
                
        except Exception as e:
            st.error(f"Error processing audio file: {str(e)}")
    else:
        st.info("ℹ️ No audio uploaded. Upload a RAVDESS file to get a prediction.")

# Text prediction column
with col2:
    if text_input and not contains_offensive_language(text_input):
        try:
            with st.spinner('Analyzing text...'):
                seq = tokenizer.texts_to_sequences([text_input])
                maxlen = text_model.input_shape[1] if hasattr(text_model.input_shape, '__len__') else 100
                padded_seq = pad_sequences(seq, maxlen=maxlen, padding='post')
                text_proba = text_model.predict(padded_seq)[0]
                text_emotion = text_labels[np.argmax(text_proba)]
                
                st.success(f"Predicted Emotion (Text): **{text_emotion.capitalize()}**", icon="💬")
                plot_emotion_probabilities(text_proba, text_labels, "Text Emotion Probabilities")
                
        except Exception as e:
            st.error(f"Error processing text input: {str(e)}")
    elif text_input:
        st.warning("Text contains offensive language - prediction may be inaccurate")

# Combined prediction if both inputs are provided
if audio_file is not None and text_input and not contains_offensive_language(text_input):
    try:
        st.subheader("Combined Prediction")
        
        # Align probabilities if the models have different label sets
        if len(audio_proba) != len(text_proba):
            st.warning("Audio and text models have different emotion sets - using text model classes")
            combined_proba = text_proba  # Fallback to text probabilities
        else:
            combined_proba = (audio_proba + text_proba) / 2
        
        combined_emotion = text_labels[np.argmax(combined_proba)]
        
        st.success(f"Combined Predicted Emotion: **{combined_emotion.capitalize()}**", icon="🤝")
        plot_emotion_probabilities(combined_proba, text_labels, "Combined Emotion Probabilities")
        
        # Debug information
        with st.expander("Debug Information"):
            st.write("Audio File Metadata:", meta)
            st.write("Text Input:", text_input)
            st.write("Audio Model Probabilities:", dict(zip(audio_labels, audio_proba)))
            st.write("Text Model Probabilities:", dict(zip(text_labels, text_proba)))
            
    except Exception as e:
        st.error(f"Error in combined prediction: {str(e)}")
else:
    if audio_file is None and not text_input:
        st.warning("Please provide audio and/or text input to get predictions.")
    elif audio_file is None:
        st.info("Provide an audio file to see combined results.")
    elif not text_input or contains_offensive_language(text_input):
        st.info("Provide clean text input to see combined results.")

# Add model information section
st.markdown("---")
st.subheader("Model Information")
st.write(f"Audio Model Classes: {', '.join(audio_labels)}")
st.write(f"Text Model Classes: {', '.join(text_labels)}")
