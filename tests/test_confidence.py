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
    # The LLM gets the band label, not the raw number (0.42 -> "low").
    assert c.format_context("I need help with my order", 0.42) == (
        'User said: "I need help with my order"\nSTT confidence: low'
    )


def test_format_uses_band_not_number():
    # No raw float should leak into the LLM-facing line.
    assert c.format_context("x", 0.9).endswith("STT confidence: high")
    assert c.format_context("x", 0.7).endswith("STT confidence: medium")
    assert c.format_context("x", 0.3).endswith("STT confidence: low")
    assert "0." not in c.format_context("x", 0.42)


def test_format_no_confidence():
    out = c.format_context("hello", None)
    assert out == 'User said: "hello"'
    assert "confidence" not in out.lower()


def test_format_empty_text():
    assert c.format_context("", 0.5) == 'User said: ""\nSTT confidence: low'
