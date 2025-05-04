import os
import joblib
import pickle
from tensorflow.keras.models import load_model

def load_audio_resources():
    """Load audio model and its encoders"""
    try:
        audio_model = load_model('full_audio_metadata_model.h5')
        
        encoders = {}
        encoder_names = ['actor', 'emotion', 'intensity', 'modality', 
                         'repetition', 'statement', 'vocal']

        for name in encoder_names:
            encoders[name] = joblib.load(f'{name}_encoder.pkl')

        print("✅ Audio model and encoders loaded successfully.")
        return audio_model, encoders
    except Exception as e:
        print("❌ Error loading audio resources:", e)
        return None, {}

def load_text_resources():
    """Load text model, tokenizer, and label encoder"""
    try:
        text_model = load_model('text_model.h5')

        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)

        label_encoder_text = joblib.load('label_encoder_text.pkl')

        print("✅ Text model resources loaded successfully.")
        return text_model, tokenizer, label_encoder_text
    except Exception as e:
        print("❌ Error loading text resources:", e)
        return None, None, None

def parse_ravdess_filename(filename):
    """Parse RAVDESS filename into metadata components"""
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    parts = name.split('-')

    if len(parts) != 7:
        raise ValueError(f"Filename '{filename}' doesn't conform to RAVDESS format")

    keys = ["modality", "vocal", "emotion", "intensity", "statement", "repetition", "actor"]
    return dict(zip(keys, parts))
