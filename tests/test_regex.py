# tests/test_regex.py

from app.regex_classifier import RegexClassifier


def test_authentication_failure():
    clf = RegexClassifier()
    label, conf = clf.predict("Failed login attempt for user admin")
    assert label == "authentication_failure"
    assert conf == 1.0


def test_authentication_success():
    clf = RegexClassifier()
    label, conf = clf.predict("User authenticated successfully")
    assert label == "authentication_success"
    assert conf == 1.0


def test_api_error():
    clf = RegexClassifier()
    label, conf = clf.predict("API error: 500 Internal Server Error")
    assert label == "api_error"
    assert conf == 1.0


def test_api_request():
    clf = RegexClassifier()
    label, conf = clf.predict("GET /v1/users")
    assert label == "api_request"
    assert conf == 1.0


def test_configuration_error():
    clf = RegexClassifier()
    label, conf = clf.predict("Invalid configuration: missing parameter")
    assert label == "configuration_error"
    assert conf == 1.0


def test_database_error():
    clf = RegexClassifier()
    label, conf = clf.predict("Database connection refused on port 5432")
    assert label == "database_error"
    assert conf == 1.0


def test_filesystem_error():
    clf = RegexClassifier()
    label, conf = clf.predict("Disk quota exceeded on /var/log")
    assert label == "filesystem_error"
    assert conf == 1.0


def test_network_error():
    clf = RegexClassifier()
    label, conf = clf.predict("DNS resolution failed for hostname")
    assert label == "network_error"
    assert conf == 1.0


def test_resource_exhaustion():
    clf = RegexClassifier()
    label, conf = clf.predict("Out of memory: process terminated")
    assert label == "resource_exhaustion"
    assert conf == 1.0


def test_security_alert():
    clf = RegexClassifier()
    label, conf = clf.predict("Unauthorized access attempt detected")
    assert label == "security_alert"
    assert conf == 1.0


def test_service_timeout():
    clf = RegexClassifier()
    label, conf = clf.predict("Request timed out after waiting 10 seconds")
    assert label == "service_timeout"
    assert conf == 1.0


def test_no_match():
    clf = RegexClassifier()
    label, conf = clf.predict("This log does not match any known rule")
    assert label is None
    assert conf == 0.0
