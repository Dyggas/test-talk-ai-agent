# test-talk-ai-agent

A real-time voice agent built on [LiveKit Agents](https://docs.livekit.io/agents/)
that is aware of speech-recognition confidence. It reads the confidence score
Deepgram attaches to every transcript and adapts its behavior: answer normally when
the transcription is reliable, and ask the user to repeat when it isn't.

```
User speech ─▶ Deepgram STT ─▶ [capture confidence] ─▶ Gemini LLM ─▶ ElevenLabs TTS ─▶ User
                                       │
                         inject "User said … / STT confidence: <band>"
                                  into the LLM context
```

## How STT confidence reaches the LLM

The whole feature is two small overrides on the `Agent` (`voice_agent.py`), plus two
pure helpers (`confidence.py`). Nothing else in the pipeline is customized.

1. **Capture — `stt_node`.** We wrap the default Deepgram STT stream. Deepgram emits a
   `SpeechEvent` of type `FINAL_TRANSCRIPT` for each finished phrase, carrying
   `alternatives[0].confidence` (a float in `[0, 1]`). We read that off the *final*
   transcript, store it on `self._last_confidence`, and re-yield every event
   unchanged so turn detection and the rest of the pipeline are untouched.

2. **Propagate — `on_user_turn_completed(turn_ctx, new_message)`.** This hook fires
   after the user's turn ends and *before* the LLM runs. We add a `system` message to
   the turn context:

   ```
   User said: "I need help with my order"
   STT confidence: high
   ```

   The LLM's system instructions describe what to do per band (high → answer; medium →
   answer but confirm; low → ask to repeat), so the model decides the behavior — the
   pipeline has no hard-coded `if confidence < 0.6` branch.

### Band label vs. raw number

The LLM is given a **qualitative band** (`high` / `medium` / `low`), not the raw float.
The thresholds live in one place (`confidence.py`):

```python
HIGH = 0.8   # >= HIGH  -> "high"
LOW  = 0.6   # <  LOW   -> "low"   (in between -> "medium")
```

Sending the band keeps the single source of truth in code and means the model never has
to compare floats near a boundary (e.g. 0.61 vs 0.59), which LLMs do unreliably. The
raw numeric value is still logged for debugging:

```
transcript='I need help with my order' confidence=0.42 band=low
```

The confidence annotation is injected into a per-turn *copy* of the chat context, so it
shapes only the current turn and never accumulates across the conversation.

---

## Setup

Requires Python 3.10+

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### API keys

Copy the template and fill in your keys:

```bash
cp .env.example .env
```

```dotenv
# LiveKit — https://cloud.livekit.io (free tier; or run the OSS server locally)
LIVEKIT_URL=wss://<your-project>.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...

# Deepgram STT — https://deepgram.com  ($200 free credit, no card)
DEEPGRAM_API_KEY=...

# Google Gemini LLM — https://aistudio.google.com/app/apikey  (free, no card)
GOOGLE_API_KEY=...

# ElevenLabs TTS — https://elevenlabs.io  (free monthly credits)
ELEVEN_API_KEY=...
```

How to obtain each (all free, no credit card required):

- **LiveKit** — create a project at [cloud.livekit.io](https://cloud.livekit.io); copy the
  URL, API key, and secret from the project settings.
- **Deepgram** — sign up at [deepgram.com](https://deepgram.com) and create an API key;
  new accounts get $200 of free credit.
- **Google Gemini** — open [Google AI Studio](https://aistudio.google.com/app/apikey),
  sign in with a Google account, and click *Create API key*.
- **ElevenLabs** — sign up at [elevenlabs.io](https://elevenlabs.io) and copy the API key
  from your profile; the free tier includes monthly credits.

`.env` is git-ignored — never commit your keys.

---

## Running

Download the bundled VAD model once, then start the agent:

```bash
# fetch the Silero VAD weights (one-time)
python voice_agent.py download-files

# quickest: talk to it in your terminal (local mic + speakers, no browser)
python voice_agent.py console

# or run as a worker and connect via the LiveKit Agents Playground
python voice_agent.py dev
```

For `dev`, open the [Agents Playground](https://agents-playground.livekit.io/), connect
it to the same LiveKit project, and join a room — the agent greets you and you can talk.

---

## Demo

`demo.py` is the quickest way to see the confidence-aware behavior. It feeds a few
hardcoded `(transcript, confidence)` pairs through the agent system prompt and
the LLM, and prints each reply. This isolates the decision logic (answer vs. ask to repeat) 
from whatever score Deepgram happens to produce, so you get a deterministic, repeatable demonstration of the high / medium / low branches. Only needs `GOOGLE_API_KEY`. To be 
honest, I really struggled to get a low score (`< 0.6`) in testing or with real audio samples,
so that will do. Real Deepgram scores are still seen when testing in your terminal.
```bash
python demo.py
```

For a live end-to-end test instead, use `python voice_agent.py console` and speak —
clearly for high confidence, mumbled or with background noise for low.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Covers the confidence band thresholds and both agent hooks (`stt_node` capture,
`on_user_turn_completed` injection) with a mocked STT stream — no live API calls.
