import pickle
import numpy as np
from tensorflow.keras.models import load_model

def load_audio_resources():
    """Load audio model and encoders"""
    audio_model = load_model('full_audio_metadata_model.h5')
    
    # Load all encoders
    encoders = {}
    for enc_name in ['actor', 'emotion', 'intensity', 'modality', 
                    'repetition', 'statement', 'vocal']:
        with open(f'{enc_name}_encoder.pkl', 'rb') as f:
            encoders[enc_name] = pickle.load(f)
    
    return audio_model, encoders

def load_text_resources():
    """Load text model resources"""
    text_model = load_model('text_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('label_encoder_text.pkl', 'rb') as f:
        label_enc = pickle.load(f)
    return text_model, tokenizer, label_enc
    
