"""
Inference script for the offensive language detection system.
Loads all three models and performs hierarchical classification.
"""

import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Constants
MAX_LEN = 100

def clean_text(text):
    """Clean and normalize text"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_models():
    """Load all models and tokenizers"""
    models = {
        'a': tf.keras.models.load_model('models/model_a.h5'),
        'b': tf.keras.models.load_model('models/model_b.h5'),
        'c': tf.keras.models.load_model('models/model_c.h5')
    }
    
    tokenizers = {}
    for model in ['a', 'b', 'c']:
        with open(f'models/tokenizer_{model}.pkl', 'rb') as f:
            tokenizers[model] = pickle.load(f)
    
    return models, tokenizers

def preprocess_text(text, tokenizer):
    """Preprocess text for model input"""
    text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    return padded

def predict_text(text):
    """
    Perform hierarchical classification on input text.
    Returns a dictionary with detailed classification results.
    """
    # Load models and tokenizers
    models, tokenizers = load_models()
    
    # Preprocess text for Model A
    text_processed = preprocess_text(text, tokenizers['a'])
    
    # Model A: Offensive vs Non-Offensive
    pred_a = models['a'].predict(text_processed)[0][0]
    is_offensive = pred_a > 0.5
    
    result = {
        'text': text,
        'is_offensive': bool(is_offensive),
        'offensive_probability': float(pred_a),
        'classification': 'OFF' if is_offensive else 'NOT',
        'targeted': None,
        'target_type': None
    }
    
    if is_offensive:
        # Model B: Targeted vs Untargeted
        text_processed = preprocess_text(text, tokenizers['b'])
        pred_b = models['b'].predict(text_processed)[0][0]
        is_targeted = pred_b > 0.5
        result['targeted'] = bool(is_targeted)
        result['targeted_probability'] = float(pred_b)
        
        if is_targeted:
            # Model C: Target Classification
            text_processed = preprocess_text(text, tokenizers['c'])
            pred_c = models['c'].predict(text_processed)[0]
            target_type = ['IND', 'GRP', 'OTH'][np.argmax(pred_c)]
            result['target_type'] = target_type
            result['target_probabilities'] = {
                'IND': float(pred_c[0]),
                'GRP': float(pred_c[1]),
                'OTH': float(pred_c[2])
            }
    
    return result

def print_prediction(result):
    """Print prediction results in a readable format"""
    print("\nOffensive Language Detection Results")
    print("=" * 40)
    print(f"Input text: {result['text']}")
    print(f"\nClassification: {result['classification']}")
    print(f"Offensive probability: {result['offensive_probability']:.2%}")
    
    if result['is_offensive']:
        print(f"\nTargeted: {'Yes' if result['targeted'] else 'No'}")
        print(f"Targeted probability: {result['targeted_probability']:.2%}")
        
        if result['targeted']:
            print(f"\nTarget type: {result['target_type']}")
            print("\nTarget probabilities:")
            for target, prob in result['target_probabilities'].items():
                print(f"- {target}: {prob:.2%}")

if __name__ == "__main__":
    # Example usage
    test_texts = [
        "Have a great day!",
        "You're an idiot",
        "Women don't belong in tech",
        "This service is terrible"
    ]
    
    print("Testing offensive language detection system...")
    for text in test_texts:
        result = predict_text(text)
        print_prediction(result)
        print("\n" + "="*50 + "\n")