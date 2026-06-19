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

    The confidence line is omitted when no score is available.
    """
    line = f'User said: "{text}"'
    if conf is None:
        return line
    return f"{line}\nSTT confidence: {conf:.2f}"
