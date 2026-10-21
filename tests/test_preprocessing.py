"""Unit tests for text cleaning and preprocessing."""
from src.data.preprocess import clean_text

def test_clean_text_basic():
    """Verify basic text cleaning: lowercasing and punctuation stripping."""
    raw = "This is a GREAT Product! 100% recommended."
    cleaned = clean_text(raw)
    assert cleaned == "this is a great product recommended"

def test_clean_text_contractions():
    """Verify that vital sentiment negations are expanded, not lost."""
    raw = "I didn't like this and won't buy again."
    cleaned = clean_text(raw)
    assert "did not" in cleaned
    assert "will not" in cleaned

def test_clean_text_html_and_urls():
    """Verify HTML tags and URLs are stripped properly."""
    raw = "Check this out <br> <a href='https://example.com'>link</a>. Awesome!"
    cleaned = clean_text(raw)
    assert "<br>" not in cleaned
    assert "https" not in cleaned
    assert "awesome" in cleaned
