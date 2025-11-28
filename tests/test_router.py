from app.router import HybridClassifier


def test_router_simple():
    clf = HybridClassifier()
    result = clf.classify("connection refused by server")

    assert "label" in result
    assert "confidence" in result
    assert result["layer"] in ["regex", "ml", "llm"]
