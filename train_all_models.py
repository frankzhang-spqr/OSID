"""
Train all three models in sequence for the hierarchical offensive language detection system.
Uses full epochs and implements additional training improvements.
"""

import os
from model_a_offensive import train_model as train_model_a
from model_b_targeted import train_model as train_model_b
from model_c_target import train_model as train_model_c
import tensorflow as tf

def set_memory_growth():
    """Configure GPU memory growth to prevent OOM errors"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def train_all():
    # Configure GPU memory growth
    set_memory_growth()
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    print("\n=== Training Model A: Offensive vs Non-Offensive ===")
    print("This model classifies text as offensive or non-offensive")
    model_a, tokenizer_a, history_a = train_model_a()
    
    print("\n=== Training Model B: Targeted vs Untargeted ===")
    print("This model determines if offensive text is targeted at someone")
    model_b, tokenizer_b, history_b = train_model_b()
    
    print("\n=== Training Model C: Target Classification ===")
    print("This model identifies the target type of offensive text")
    model_c, tokenizer_c, history_c = train_model_c()
    
    print("\nAll models have been trained and saved in the 'models' directory.")
    print("\nSaved files:")
    print("- Model A: model_a.h5, tokenizer_a.pkl")
    print("  Purpose: Offensive vs Non-Offensive classification")
    print("  Training completed with final validation accuracy: {:.2%}".format(
        max(history_a.history['val_accuracy'])
    ))
    
    print("\n- Model B: model_b.h5, tokenizer_b.pkl")
    print("  Purpose: Targeted vs Untargeted classification")
    print("  Training completed with final validation accuracy: {:.2%}".format(
        max(history_b.history['val_accuracy'])
    ))
    
    print("\n- Model C: model_c.h5, tokenizer_c.pkl")
    print("  Purpose: Target type classification (IND/GRP/OTH)")
    print("  Training completed with final validation accuracy: {:.2%}".format(
        max(history_c.history['val_accuracy'])
    ))
    
    print("\nTo make predictions using these models, use predict_offensive.py")

if __name__ == "__main__":
    train_all()