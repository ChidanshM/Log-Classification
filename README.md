# Hybrid Log Classification System

This project implements a hybrid approach to log classification using three sequential layers:

1. Rule-Based Regex Classifier  
2. Machine Learning Classifier (Sentence Embeddings + Logistic Regression)  
3. Large Language Model (LLM) Fallback Classifier  

The system is designed for high-volume and diverse log environments where traditional single-method classification is insufficient.

---

## Architecture Overview

```mermaid
flowchart LR

    subgraph Ingestion["Data Ingestion and Parsing"]
        A[Raw Logs]
        B[Drain3 Parser]
        C[Parsed Template]
    end

    subgraph Hybrid["Hybrid Classification Core"]
        D[Regex Rules]
        E[Sentence Embeddings]
        F[Logistic Regression]
        G[LLM Fallback]
        H[Confidence Router]
    end

    subgraph API["API Layer"]
        L[FastAPI Endpoint]
        M[CSV Upload]
        N[JSON/CSV Output]
    end

    A --> B --> C --> H
    H --> D -->|High| N
    D -->|No Match| E --> F --> H
    H -->|ML High| N
    H -->|ML Low| G --> N

    M --> L --> C
    N --> L
---
Project Structure 


hybrid_log_classifier/
│
├── app/
│   ├── api.py
│   ├── config.py
│   ├── log_parser.py
│   ├── regex_classifier.py
│   ├── ml_classifier.py
│   ├── llm_classifier.py
│   ├── router.py
│   ├── models.py
│
├── scripts/
│   ├── train_ml.py
│   ├── evaluate.py
│
├── models/
│   ├── lr_model.pkl
│   ├── label_encoder.pkl
│
├── tests/
│   ├── test_regex.py
│   ├── test_ml.py
│   ├── test_router.py
│   ├── test_api.py
│
├── requirements.txt
├── README.md
└── run_api.sh

Installation

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt


