"""Confidence-aware LiveKit voice agent.

Pipeline: user speech -> Deepgram STT -> Google Gemini LLM -> ElevenLabs TTS -> user.

The agent reads the confidence score Deepgram attaches to each transcript and adapts
its behavior (answer normally on high confidence, ask the user to repeat on low
confidence). The confidence-capture and context-injection hooks are added on top of
this skeleton in the following commits.
"""

import logging
from typing import AsyncIterable

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import Agent, AgentSession, ModelSettings, stt
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.plugins import deepgram, elevenlabs, google

from confidence import confidence_band, format_context

load_dotenv()

logger = logging.getLogger("confidence-agent")


class ConfidenceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful, friendly voice assistant.\n"
                "Each user message is annotated with an STT confidence level — one of "
                "high, medium, or low — indicating how reliably their speech was "
                "transcribed.\n"
                "- high: answer normally.\n"
                "- low: do NOT assume what was said. Politely ask the user to repeat or "
                'rephrase (e.g. "Sorry, I may have misheard you — could you say that '
                'again?").\n'
                "- medium: answer, but briefly confirm your understanding.\n"
                "Keep replies concise and conversational. Never mention the confidence "
                "level aloud."
            )
        )
        self._last_confidence: float | None = None

    async def stt_node(
        self,
        audio: AsyncIterable[rtc.AudioFrame],
        model_settings: ModelSettings,
    ) -> AsyncIterable[stt.SpeechEvent | str]:
        """Wrap the default Deepgram STT stream to capture the confidence score
        from an STT event and re-yield it.
        """
        async for event in Agent.default.stt_node(self, audio, model_settings):
            if (
                isinstance(event, stt.SpeechEvent)
                and event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
                and event.alternatives
            ):
                self._last_confidence = event.alternatives[0].confidence
            yield event

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        """Inject the transcript + last captured confidence into the LLM context."""
        if self._last_confidence is None:
            return  # no final transcript captured yet

        transcript = new_message.text_content or ""
        turn_ctx.add_message(
            role="system",
            content=format_context(transcript, self._last_confidence),
        )
        logger.info(
            "transcript=%r confidence=%.2f band=%s",
            transcript,
            self._last_confidence,
            confidence_band(self._last_confidence),
        )


async def entrypoint(ctx: agents.JobContext) -> None:
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=elevenlabs.TTS(),
        # We always rewrite the chat context in on_user_turn_completed (to add the
        # confidence line), which invalidates any speculative draft — so preemptive
        # generation would just be discarded every turn. Turn it off.
        preemptive_generation=False,
    )
    await session.start(room=ctx.room, agent=ConfidenceAgent())
    await session.generate_reply(
        instructions="Greet the user warmly and ask how you can help."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
