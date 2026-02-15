# News Classification with GRUs & Bi-Directional LSTMs

An advanced NLP project that classifies news articles into specific topics using Gated Recurrent Units (GRUs) and Bi-Directional LSTMs.

Building on standard RNNs/LSTMs, this project explores two critical optimizations in sequence modeling: Efficiency (using GRUs) and Contextual Depth (using Bi-Directional processing). It demonstrates how to achieve state-of-the-art performance on text classification tasks while managing computational complexity.
## Objective

The goal was to improve upon standard LSTM classifiers by testing architectures that are either faster or smarter.

  1. The Problem: Standard LSTMs can be computationally heavy, and unidirectional models often miss context that appears later in a sentence.
  2. The Solution:
        * GRUs: Use Gated Recurrent Units, which simplify the LSTM cell (removing the separate cell state) to reduce parameter count and training time without sacrificing significant accuracy.
        * Bi-Directional LSTMs: Process text from both start-to-end and end-to-start simultaneously, allowing the network to understand a word based on both its past and future context.
  3. The Takeaway: GRUs offer the best "bang for your buck" in terms of speed/accuracy, while Bi-Directional models excel when maximum context is required.

## Key Concepts & Skills

* Gated Recurrent Units (GRUs): Implementing streamlined RNN cells that control information flow using only Update and Reset gates (vs. LSTM's three gates).
* Bi-Directional Processing: Wrapping RNN layers to read input sequences in both directions, doubling the context available to the classification head.
* Stacked Architectures: Building multi-layer (Double) GRU networks to capture increasingly abstract features from the text.
* Hyperparameter Tuning: Managing dropout rates (dropout vs recurrent_dropout) to prevent overfitting in deep recurrent networks.

## Methodology / Architecture
1. Text Preprocessing

    * Tokenization & Embedding: Converting news titles/descriptions into integer sequences and passing them through a trainable Embedding layer to learn dense vector representations.
    * Dimensionality: Input sequences standardized to a fixed length (e.g., 12 words) with an embedding dimension of 100.

2. Model Variants Implemented

I designed and compared three distinct architectures:

* Single-Layer GRU: A lightweight baseline focused on maximum inference speed.
* Double-Layer GRU: A stacked architecture where the first GRU returns full sequences (return_sequences=True) to feed the second, enabling deeper feature extraction.
* Bi-Directional LSTM: A powerful model that processes text forwards and backwards to capture full semantic context.

3. Training & Optimization

    * Optimizer: Adam (learning_rate=0.001).
    * Regularization: Applied EarlyStopping and ModelCheckpoint to save the best weights and prevent overfitting.
    * Loss Function: Sparse Categorical Crossentropy for multi-class classification (8 distinct news topics).

## Code Highlight

Here is the implementation of the Bi-Directional LSTM, which combines the memory capacity of LSTMs with dual-direction processing:
```Python

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

def create_bidirectional_lstm(vocab_size, embedding_dim=100, input_length=12, num_classes=8):
    model = Sequential([
        # 1. Vectorize text
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
        
        # 2. Process forwards AND backwards
        # 128 units total (64 forward + 64 backward usually, or 128 each depending on impl)
        Bidirectional(LSTM(units=128, 
                           activation='tanh',
                           dropout=0.3)), 
        
        # 3. Classify
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model
```
## Results

The project highlights the trade-offs between different RNN architectures:

* GRUs: Achieved comparable accuracy to standard LSTMs but converged faster due to having fewer parameters.
* Bi-Directional LSTM: Provided the highest classification accuracy by effectively handling ambiguous headlines where context from the end of the sentence resolved meanings at the beginning.
* Stability: The use of recurrent_dropout significantly reduced overfitting on the relatively short news headlines.

(_Further comments in the python notebook_)

<img width="916" height="591" alt="image" src="https://github.com/user-attachments/assets/8ea543e7-a8af-446a-a315-83e0843ed727" />
<img width="907" height="591" alt="image" src="https://github.com/user-attachments/assets/11a35e77-0bea-41bc-b87d-f58f27467c4b" />


## Dependencies

```Python 3.x
    TensorFlow / Keras
    Gensim (for optional advanced preprocessing)
    Pandas & Matplotlib
```

## How to Run

Clone the repository.
Download the Labeled Newscatcher Dataset (or similar news dataset) and place it in the input directory.
Run news-classification-using-grus.ipynb.
The notebook will train all three models (Single GRU, Double GRU, Bi-LSTM) and plot a comparison of their accuracy.

## Future Improvements

Attention Mechanism: Implement a custom Attention layer on top of the Bi-LSTM to visualize which words the model focuses on for classification.
Transformer Baseline: Compare these recurrent models against a small BERT-tiny or DistilBERT model to see if the transformer overhead is worth the performance gain.

## References / Credits

Dataset: [Labeled Newscatcher Dataset](https://www.kaggle.com/datasets/kotartemiy/topic-labeled-news-dataset)
Concept: "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" (Chung et al., 2014)
