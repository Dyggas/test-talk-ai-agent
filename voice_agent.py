"""Confidence-aware LiveKit voice agent.

Pipeline: user speech -> Deepgram STT -> Google Gemini LLM -> ElevenLabs TTS -> user.

The agent reads the confidence score Deepgram attaches to each transcript and adapts
its behavior (answer normally on high confidence, ask the user to repeat on low
confidence). The confidence-capture and context-injection hooks are added on top of
this skeleton in the following commits.
"""

import logging

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession
from livekit.plugins import deepgram, elevenlabs, google

from confidence import HIGH, LOW

load_dotenv()

logger = logging.getLogger("confidence-agent")


class ConfidenceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful, friendly voice assistant.\n"
                "Each user message is annotated with an STT confidence score in [0, 1] "
                "indicating how reliably their speech was transcribed.\n"
                f"- confidence >= {HIGH}: answer normally.\n"
                f"- confidence < {LOW}: do NOT assume what was said. Politely ask the user "
                'to repeat or rephrase (e.g. "Sorry, I may have misheard you — could you '
                'say that again?").\n'
                "- in between: answer, but briefly confirm your understanding.\n"
                "Keep replies concise and conversational. Never read the score aloud."
            )
        )
        self._last_confidence: float | None = None


async def entrypoint(ctx: agents.JobContext) -> None:
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=elevenlabs.TTS(),
    )
    await session.start(room=ctx.room, agent=ConfidenceAgent())
    await session.generate_reply(
        instructions="Greet the user warmly and ask how you can help."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
