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
