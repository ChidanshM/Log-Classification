#!/usr/bin/env python3
"""
Quick test script for Regex + ML layers (no LLM required)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.regex_classifier import RegexClassifier
from app.ml_classifier import MLClassifier

# Test logs covering all 11 categories
test_logs = [
    "Failed login attempt for user 'admin'",
    "User 'john' successfully authenticated",
    "API error 500 for endpoint /v1/process",
    "GET /v1/metrics from client",
    "Out of memory: process 'worker-1' killed",
    "Security alert: unauthorized access attempt",
    "Network unreachable while contacting 10.0.0.1",
    "Request timed out contacting 'auth-service'",
    "Database connection refused on 10.0.0.5",
    "Configuration error: missing key 'cluster.id'",
    "Filesystem error: disk quota exceeded at '/var/log'",
]

def test_classifiers():
    print("\n" + "="*80)
    print("🧪 TESTING ENHANCED CLASSIFIERS (Regex + ML)")
    print("="*80 + "\n")
    
    regex_clf = RegexClassifier()
    ml_clf = MLClassifier()
    
    for i, log in enumerate(test_logs, 1):
        print(f"\n{i}. Log: {log[:70]}{'...' if len(log) > 70 else ''}")
        print("-" * 80)
        
        # Test Regex
        regex_label, regex_conf = regex_clf.predict(log)
        if regex_label:
            print(f"   🔍 REGEX: {regex_label} (confidence: {regex_conf:.2%})")
        else:
            print(f"   🔍 REGEX: No match")
        
        # Test ML
        try:
            ml_label, ml_conf = ml_clf.predict(log)
            print(f"   🤖 ML:    {ml_label} (confidence: {ml_conf:.2%})")
        except Exception as e:
            print(f"   🤖 ML:    Error - {e}")
    
    print("\n" + "="*80)
    print("✅ Testing Complete!")
    print("="*80 + "\n")
    
    print("📝 Summary:")
    print("   - Regex classifier uses pattern matching (fast, deterministic)")
    print("   - ML classifier uses learned embeddings + features (probabilistic)")
    print("   - For the HDFS log, it will likely be classified by ML as the")
    print("     regex patterns don't cover it yet.")
    print()

if __name__ == "__main__":
    test_classifiers()
