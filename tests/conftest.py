import pytest
from livekit.agents import Agent, stt

import voice_agent


def make_speech_event(event_type, confidence=None, text="hi", with_alternatives=True):
    """Build a real SpeechEvent with one alternative (or none)."""
    alternatives = []
    if with_alternatives:
        alternatives = [
            stt.SpeechData(
                language="en",
                text=text,
                start_time=0.0,
                end_time=1.0,
                confidence=confidence,
            )
        ]
    return stt.SpeechEvent(type=event_type, alternatives=alternatives)


@pytest.fixture
def speech_event():
    return make_speech_event


@pytest.fixture
def agent():
    return voice_agent.ConfidenceAgent()


@pytest.fixture
def drive_stt_node(monkeypatch):
    """Run ``agent.stt_node`` with the default Deepgram stream replaced by ``events``.

    Returns the list of events the override re-yielded (to assert pass-through).
    """

    async def _run(agent, events):
        async def fake_default(self, audio, model_settings):
            for event in events:
                yield event

        monkeypatch.setattr(Agent.default, "stt_node", fake_default)
        collected = []
        async for event in agent.stt_node(audio=None, model_settings=None):
            collected.append(event)
        return collected

    return _run


class FakeChatContext:
    """Records ``add_message`` calls so tests can assert what was injected."""

    def __init__(self):
        self.added = []

    def add_message(self, *, role, content, **kwargs):
        self.added.append({"role": role, "content": content})
        return content


class FakeChatMessage:
    def __init__(self, text):
        self._text = text

    @property
    def text_content(self):
        return self._text


@pytest.fixture
def chat_context():
    return FakeChatContext


@pytest.fixture
def chat_message():
    return FakeChatMessage
