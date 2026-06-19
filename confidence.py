# Behavior thresholds:
#   >= HIGH  -> answer normally
#   <  LOW   -> ask the user to repeat / clarify
#   between  -> answer, but confirm understanding
HIGH = 0.8
LOW = 0.6


def confidence_band(conf: float | None) -> str:
    """Classify a confidence score into a behavior band.

    Returns ``"high"`` (>= HIGH), ``"low"`` (< LOW), ``"medium"`` (in between),
    or ``"unknown"`` when no score is available yet.
    """
    if conf is None:
        return "unknown"
    if conf >= HIGH:
        return "high"
    if conf < LOW:
        return "low"
    return "medium"


def format_context(text: str, conf: float | None) -> str:
    """Build the context line injected into the LLM before it replies.

    The LLM is given the qualitative band ("high"/"medium"/"low") rather than the
    raw float, so it never has to compare numbers near the thresholds::

        User said: "I need help with my order"
        STT confidence: low

    The raw numeric value is kept for the logs (see voice_agent). The confidence
    line is omitted when no score is available.
    """
    line = f'User said: "{text}"'
    if conf is None:
        return line
    return f"{line}\nSTT confidence: {confidence_band(conf)}"
