import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import re
import os

# Constants
MAX_WORDS = 15000
MAX_LEN = 100
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2

def clean_text(text):
    """Clean and normalize text"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_data():
    """Load and preprocess the training data for Model C"""
    train_df = pd.read_csv('data/olid-training-v1.0.tsv', sep='\t')
    
    # Filter only targeted offensive tweets
    targeted_df = train_df[
        (train_df['subtask_a'] == 'OFF') & 
        (train_df['subtask_b'] == 'TIN')
    ].copy()
    
    targeted_df['tweet'] = targeted_df['tweet'].apply(clean_text)
    
    # Map labels for Model C (IND vs GRP vs OTH)
    label_map = {'IND': 0, 'GRP': 1, 'OTH': 2}
    targeted_df['label'] = targeted_df['subtask_c'].map(label_map)
    
    # Remove rows with NULL labels
    targeted_df = targeted_df.dropna(subset=['label'])
    
    return targeted_df[['tweet', 'label']]

def preprocess_text(texts, tokenizer=None):
    """Tokenize and pad the text data"""
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<UNK>')
        tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    return padded_sequences, tokenizer

def create_model(vocab_size, num_classes=3):
    """Create Model C for target classification"""
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Multi-class classification
    ])
    
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=1e-3,
        weight_decay=1e-5
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Train Model C"""
    print("Loading data...")
    train_df = load_data()
    
    print("Preprocessing text...")
    X_padded, tokenizer = preprocess_text(train_df['tweet'])
    y = train_df['label'].values
    
    # Split data with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        X_padded, y,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=y
    )
    
    print("Creating model...")
    model = create_model(min(len(tokenizer.word_index) + 1, MAX_WORDS))
    
    # Class weights for imbalanced dataset
    class_weights = dict(zip(
        np.unique(y_train),
        1 / np.bincount(y_train) * len(y_train) / 3
    ))
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            'models/model_c.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    print("Training Model C (Target Classification: IND vs GRP vs OTH)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    print("\nEvaluating Model C...")
    y_pred = model.predict(X_val)
    y_pred = np.argmax(y_pred, axis=1)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['IND', 'GRP', 'OTH']))
    
    # Save tokenizer
    import pickle
    with open('models/tokenizer_c.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    return model, tokenizer, history

if __name__ == "__main__":
    model, tokenizer, history = train_model()