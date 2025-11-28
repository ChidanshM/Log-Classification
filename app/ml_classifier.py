# app/ml_classifier.py

import pickle
from typing import Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from app.config import settings


# Load the embedding model ONCE (faster)
_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class MLClassifier:
    """
    ML classifier layer:
    - Converts text into embeddings
    - Predicts label + confidence using trained logistic regression
    """

    def __init__(self):
        # Load logistic regression model
        with open(settings.LR_MODEL_PATH, "rb") as f:
            self.model: LogisticRegression = pickle.load(f)

        # Load label encoder
        with open(settings.LABEL_ENCODER_PATH, "rb") as f:
            self.label_encoder: LabelEncoder = pickle.load(f)

        # Reuse global encoder
        self.encoder = _encoder

    def predict(self, text: str) -> Tuple[Optional[str], float]:
        """
        Returns:
            label: str
            confidence: float (0.0 - 1.0)
        """
        embedding = self.encoder.encode([text])  # shape = (1, 384)

        # Sklearn predict_proba requires 2D numpy array
        embedding_np = np.array(embedding)

        # Predict probabilities
        probs = self.model.predict_proba(embedding_np)[0]

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        label = self.label_encoder.inverse_transform([best_idx])[0]

        return label, confidence
