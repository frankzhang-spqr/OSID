import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import re

# Constants
MAX_WORDS = 15000  # Increased vocabulary size
MAX_LEN = 100      # Increased sequence length
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2

def clean_text(text):
    """Clean and normalize text"""
    text = text.lower()
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_data():
    """Load and preprocess the training data"""
    # Read training data
    train_df = pd.read_csv('data/olid-training-v1.0.tsv', sep='\t')
    
    # Clean text
    train_df['tweet'] = train_df['tweet'].apply(clean_text)
    
    # Map labels to numbers
    label_map = {'NOT': 0, 'OFF': 1}
    train_df['subtask_a'] = train_df['subtask_a'].map(label_map)
    
    return train_df

def preprocess_text(texts, tokenizer=None):
    """Tokenize and pad the text data"""
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<UNK>')
        tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    return padded_sequences, tokenizer

def create_model(vocab_size):
    """Create the neural network model"""
    model = tf.keras.Sequential([
        # Input embedding layer
        tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
        
        # Convolutional layers for feature extraction
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        
        # Bidirectional LSTM layers
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        
        # Dense layers with strong regularization
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Use AdamW optimizer with weight decay
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=1e-3,
        weight_decay=1e-5
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Train the offensive language detection model"""
    # Load data
    print("Loading data...")
    train_df = load_data()
    
    # Preprocess text
    print("Preprocessing text...")
    X_padded, tokenizer = preprocess_text(train_df['tweet'])
    y = train_df['subtask_a'].values
    
    # Split data with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        X_padded, y, 
        test_size=VALIDATION_SPLIT, 
        random_state=42,
        stratify=y
    )
    
    # Create model
    print("Creating model...")
    model = create_model(min(len(tokenizer.word_index) + 1, MAX_WORDS))
    
    # Class weights for imbalanced dataset
    class_weights = dict(zip(
        np.unique(y_train),
        1 / np.bincount(y_train) * len(y_train) / 2
    ))
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            'models/offensive_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_format='h5'
        )
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred = (model.predict(X_val) > 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['NOT', 'OFF']))
    
    return model, tokenizer, history

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    
    # Train the model
    model, tokenizer, history = train_model()
    
    # Save tokenizer
    import pickle
    with open('models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)