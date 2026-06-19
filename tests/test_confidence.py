import confidence as c

# --- confidence_band ---------------------------------------------------------


def test_band_high_boundary():
    assert c.confidence_band(0.8) == "high"
    assert c.confidence_band(0.95) == "high"


def test_band_low_boundary():
    assert c.confidence_band(0.59) == "low"
    assert c.confidence_band(0.6) == "medium"


def test_band_medium():
    assert c.confidence_band(0.6) == "medium"
    assert c.confidence_band(0.79) == "medium"


def test_band_unknown():
    assert c.confidence_band(None) == "unknown"


# --- format_context ----------------------------------------------------------


def test_format_with_confidence():
    assert c.format_context("I need help with my order", 0.42) == (
        'User said: "I need help with my order"\nSTT confidence: 0.42'
    )


def test_format_rounding():
    assert c.format_context("x", 0.4249).endswith("STT confidence: 0.42")
    assert c.format_context("x", 0.4).endswith("STT confidence: 0.40")


def test_format_no_confidence():
    out = c.format_context("hello", None)
    assert out == 'User said: "hello"'
    assert "confidence" not in out.lower()


def test_format_empty_text():
    assert c.format_context("", 0.5) == 'User said: ""\nSTT confidence: 0.50'
