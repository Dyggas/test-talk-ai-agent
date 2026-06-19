"""Deterministic demo of the confidence-aware behavior — no audio, STT, TTS, or room.

Feeds hardcoded (transcript, confidence) pairs through the *real* agent system prompt
and the *real* Gemini LLM, exactly as on_user_turn_completed would, and prints the reply.
This isolates the interesting question — does the agent answer vs. ask to repeat based on
confidence — from whether Deepgram happens to produce a given score.

Only needs GOOGLE_API_KEY in .env.

    python demo.py
"""

import asyncio

from dotenv import load_dotenv
from livekit.agents import llm
from livekit.plugins import google

from confidence import confidence_band, format_context
from voice_agent import ConfidenceAgent

# Hardcoded interactions to demo: (what the user "said", STT confidence).
INTERACTIONS = [
    ("What are your opening hours?", 0.97),     # high  -> answer normally
    ("Can you change my delivery address?", 0.72),  # medium -> answer + confirm
    ("I need help with my order", 0.41),        # low   -> ask to repeat
]


async def main() -> None:
    load_dotenv()
    gllm = google.LLM(model="gemini-2.5-flash-lite")
    instructions = ConfidenceAgent().instructions  # reuse the real system prompt

    for text, conf in INTERACTIONS:
        ctx = llm.ChatContext.empty()
        ctx.add_message(role="system", content=instructions)
        ctx.add_message(role="system", content=format_context(text, conf))  # the injection
        ctx.add_message(role="user", content=text)

        reply = ""
        async for piece in gllm.chat(chat_ctx=ctx).to_str_iterable():
            reply += piece

        print(f"\n[{confidence_band(conf)}  confidence={conf:.2f}]")
        print(f"  user : {text}")
        print(f"  agent: {reply.strip()}")

    await gllm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
