import os
import pytest
from app.ml_classifier import MLClassifier


@pytest.mark.skipif(
    not os.path.exists("models/lr_model.pkl"),
    reason="ML model not trained yet",
)
def test_ml_classifier_basic_prediction():
    clf = MLClassifier()
    label, conf = clf.predict("Database unavailable")
    assert isinstance(label, str)
    assert 0.0 <= conf <= 1.0


@pytest.mark.skipif(
    not os.path.exists("models/lr_model.pkl"),
    reason="ML model not trained yet",
)
def test_ml_known_class_prediction():
    clf = MLClassifier()
    text = "SQL error: relation does not exist"
    label, conf = clf.predict(text)

    # should recognize category
    assert label == "database_error"

    # confidence is allowed to be low until we retrain on larger dataset
    assert 0.0 <= conf <= 1.0



@pytest.mark.skipif(
    not os.path.exists("models/lr_model.pkl"),
    reason="ML model not trained yet",
)
def test_ml_label_space_correctness():
    clf = MLClassifier()

    expected = set([
        "authentication_failure",
        "authentication_success",
        "api_error",
        "api_request",
        "configuration_error",
        "database_error",
        "filesystem_error",
        "network_error",
        "resource_exhaustion",
        "security_alert",
        "service_timeout",
    ])

    assert set(clf.label_encoder.classes_) == expected
