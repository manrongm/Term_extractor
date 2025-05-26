# TermExtractor: A Modular Pipeline for Scientific Terminology Extraction

**TermExtractor** is a modular system for extracting domain-specific terminology from academic texts. Designed for adaptability across scientific disciplines, it combines rule-based preprocessing with a deep learning architecture to identify and label technical terms in context.

## 📌 Features

- End-to-end processing from PDF to tagged sequences.
- Linguistic preprocessing using spaCy.
- Fuzzy matching and BIO tag generation.
- CRF-based sequence labeling with BERT + BiLSTM backbone.
- Support for warm-start fine-tuning to combat overfitting.

## 🧱 Pipeline Overview

1. **Text Extraction**: Converts scientific PDF documents to plain text using `pdfminer`.
2. **Preprocessing**: Cleans and tokenizes text; extracts noun phrases via spaCy.
3. **BIO Tagging**: Applies fuzzy matching and assigns BIO labels to tokens.
4. **Data Cleaning**: Filters out noisy or generic terms, enforces BIO consistency.
5. **Tokenization & Encoding**: Converts text into BERT-compatible inputs.
6. **Model**: Uses a BERT + BiLSTM + CRF architecture for sequence labeling.
7. **Training**: Supports warm-start training, dropout regularization, and early stopping.

## 🏁 Final Model

The best-performing configuration uses:
- BERT encoder (non-frozen)
- 1-layer BiLSTM
- CRF decoder
- Warm-start from previously trained checkpoints

## 📊 Evaluation

| Method                      | Precision | Recall | F1 Score |
|----------------------------|-----------|--------|----------|
| TF-IDF + C-Value           | 0.502     | 0.621  | 0.555    |
| DistilBERT-BiLSTM-CRF      | 0.611     | 0.625  | 0.618    |
| SciBERT-BiLSTM-CRF         | 0.628     | 0.595  | 0.611    |
| Frozen BERT-BiLSTM-CRF     | 0.660     | 0.622  | 0.640    |
| BERT-BiLSTM-CRF (concat)   | 0.657     | 0.697  | 0.676    |
| BERT-BiLSTM-CRF            | 0.689     | 0.665  | 0.677    |
| **BERT-BiLSTM-CRF + Warm** | **0.741** | **0.707** | **0.724** |

## 📂 Directory Structure

```
.
├── extract.py                 # PDF to TXT conversion
├── preprocess.py             # Cleaning, sentence and phrase segmentation
├── generate_bio_tags.py      # Tagging tokens using noun phrases
├── clean_data_bio.py         # Heuristic-based data cleaning
├── preprocess_bert_data.py   # BERT-compatible input generation
├── bert_bilstm_crf.py        # Model definition
├── train_bert.py             # Training loop
├── test_model.py             # Evaluation script
└── README.md
```

## 📄 License

MIT License
