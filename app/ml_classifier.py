# app/ml_classifier.py

import pickle
from typing import Tuple, Optional
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from app.config import settings
from app.feature_extractor import LogFeatureExtractor


# Load the embedding model ONCE (faster)
_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
_feature_extractor = LogFeatureExtractor()


class MLClassifier:
    """
    ML classifier layer:
    - Converts text into sentence embeddings (semantic features)
    - Extracts domain-specific features (structural/pattern features)
    - Combines both feature types for improved classification
    - Predicts label + confidence using trained model
    
    Improvements:
    - Feature engineering with domain-specific patterns
    - Combined embeddings + handcrafted features
    - Better confidence calibration
    """

    def __init__(self):
        # Load logistic regression model
        with open(settings.LR_MODEL_PATH, "rb") as f:
            self.model: LogisticRegression = pickle.load(f)

        # Load label encoder
        with open(settings.LABEL_ENCODER_PATH, "rb") as f:
            self.label_encoder: LabelEncoder = pickle.load(f)

        # Reuse global encoder and feature extractor
        self.encoder = _encoder
        self.feature_extractor = _feature_extractor
        
        # Check if we have a combined feature model
        self.use_combined_features = hasattr(self.model, 'n_features_in_') and \
                                     self.model.n_features_in_ > 384

    def predict(self, text: str) -> Tuple[Optional[str], float]:
        """
        Predict log category with enhanced features.
        
        Returns:
            label: str (category name)
            confidence: float (0.0 - 1.0)
        """
        # Get sentence embedding (384 dimensions)
        embedding = self.encoder.encode([text])  # shape = (1, 384)
        embedding_np = np.array(embedding)
        
        # Extract handcrafted features if model expects them
        if self.use_combined_features:
            handcrafted_features = self.feature_extractor.extract_features(text)
            handcrafted_features = handcrafted_features.reshape(1, -1)
            
            # Combine embeddings with handcrafted features
            combined_features = np.hstack([embedding_np, handcrafted_features])
        else:
            # Fallback to embeddings only (for backward compatibility)
            combined_features = embedding_np

        # Predict probabilities
        probs = self.model.predict_proba(combined_features)[0]

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        label = self.label_encoder.inverse_transform([best_idx])[0]

        return label, confidence

