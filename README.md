# Offensive Language Detection System

This directory contains trained models and tokenizers for the hierarchical offensive language detection system.

## Model Architecture

The system consists of three hierarchical models:

1. Model A (`model_a.h5`, `tokenizer_a.pkl`)
   - Task: Offensive vs Non-Offensive classification
   - Input: Text
   - Output: NOT (Not Offensive) or OFF (Offensive)
   - Architecture: CNN-BiLSTM with attention

2. Model B (`model_b.h5`, `tokenizer_b.pkl`)
   - Task: Targeted vs Untargeted classification
   - Input: Offensive text only
   - Output: TIN (Targeted) or UNT (Untargeted)
   - Architecture: CNN-BiLSTM with attention

3. Model C (`model_c.h5`, `tokenizer_c.pkl`)
   - Task: Target classification
   - Input: Targeted offensive text only
   - Output: IND (Individual), GRP (Group), or OTH (Other)
   - Architecture: CNN-BiLSTM with attention

## Usage

To use the models for inference, use the `predict_offensive.py` script:

```python
from predict_offensive import predict_text

text = "Your input text here"
result = predict_text(text)
print(result)
```

## Model Performance

Current model performance metrics:

### Model A
- Accuracy: 76%
- Precision: 79% (NOT), 69% (OFF)
- Recall: 88% (NOT), 52% (OFF)

### Model B
- Accuracy: 83%
- F1-Score: 91% (TIN), 21% (UNT)

### Model C
- Accuracy: 64%
- F1-Score: 77% (IND), 45% (GRP), 19% (OTH)

## File Structure
```
models/
├── model_a.h5          # Model A weights
├── model_b.h5          # Model B weights
├── model_c.h5          # Model C weights
├── tokenizer_a.pkl     # Tokenizer for Model A
├── tokenizer_b.pkl     # Tokenizer for Model B
└── tokenizer_c.pkl     # Tokenizer for Model C