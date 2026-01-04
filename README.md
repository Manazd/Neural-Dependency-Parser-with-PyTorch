# Neural-Dependency-Parser-with-PyTorch


## 📖 Overview

This repository implements a **transition-based neural dependency parser** using PyTorch. The project provides an end-to-end pipeline for training and evaluating a dependency parser on **CoNLL-formatted treebank data**, following a shift–reduce style parsing framework.

The parser learns to predict parsing actions from parser configurations using a feedforward neural network trained on extracted features.

---

## Key Components

- **CoNLL data reader** for dependency treebanks
- **Feature extraction** from parser configurations
- **Vocabulary construction** for words and POS tags
- **Neural transition classifier** implemented in PyTorch
- **Greedy decoder** for dependency parsing
- **Evaluation script** for measuring parsing accuracy

---

## Project Structure

- `conll_reader.py` – Reads and represents CoNLL dependency data  
- `extract_training_data.py` – Extracts parser states and gold actions  
- `get_vocab.py` – Builds vocabularies from training data  
- `train_model.py` – Trains the neural dependency parser  
- `decoder.py` – Greedy transition-based decoding  
- `evaluate.py` – Runs evaluation on test data  
- `words.vocab` – Generated vocabulary file  
- `model.pt` – Trained model checkpoint  

---

## Model

The parser uses a **feedforward neural network** with:
- Learned word embeddings
- Hidden layers with ReLU activation
- Softmax-based action prediction

Training is performed using supervised learning over gold transition sequences derived from dependency trees.

---

## Requirements

- Python 3
- PyTorch
- NumPy

---

## License

This project is for academic and educational use.
