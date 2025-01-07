"""Tests for text normalization service"""

import pytest
from api.src.services.text_processing.normalizer import normalize_text

def test_urls():
    """Test URL handling"""
    # URLs with http/https
    assert normalize_text("Check out https://example.com") == "Check out http example dot com"
    assert normalize_text("Visit http://site.com/docs") == "Visit http site dot com slash docs"
    
    # URLs with www
    assert normalize_text("Go to www.example.com") == "Go to www example dot com"
    
    # Email addresses
    assert normalize_text("Email me at user@example.com") == "Email me at user at example dot com"
    
    # Normal text should be unaffected, other than downstream normalization
    assert normalize_text("This is not.a.url text") == "This is not-a-url text"
    assert normalize_text("Hello, how are you today?") == "Hello, how are you today?"
    assert normalize_text("It costs $50.") == "It costs 50 dollars."
