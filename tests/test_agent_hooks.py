from livekit.agents import stt

import confidence as c

FINAL = stt.SpeechEventType.FINAL_TRANSCRIPT
INTERIM = stt.SpeechEventType.INTERIM_TRANSCRIPT


# --- stt_node: capture confidence -------------------------------------------


async def test_stt_node_captures_final_confidence(agent, drive_stt_node, speech_event):
    await drive_stt_node(agent, [speech_event(FINAL, confidence=0.91)])
    assert agent._last_confidence == 0.91


async def test_stt_node_ignores_interim(agent, drive_stt_node, speech_event):
    await drive_stt_node(agent, [speech_event(INTERIM, confidence=0.5)])
    assert agent._last_confidence is None


async def test_stt_node_passthrough(agent, drive_stt_node, speech_event):
    events = [
        speech_event(INTERIM, confidence=0.5),
        speech_event(FINAL, confidence=0.9),
        "another-event",
    ]
    out = await drive_stt_node(agent, events)
    assert out == events


async def test_stt_node_no_alternatives(agent, drive_stt_node, speech_event):
    await drive_stt_node(agent, [speech_event(FINAL, with_alternatives=False)])
    assert agent._last_confidence is None


async def test_stt_node_keeps_latest(agent, drive_stt_node, speech_event):
    await drive_stt_node(
        agent,
        [speech_event(FINAL, confidence=0.4), speech_event(FINAL, confidence=0.95)],
    )
    assert agent._last_confidence == 0.95


# --- on_user_turn_completed: inject into context ----------------------------


async def test_turn_injects_transcript_and_confidence(
    agent, chat_context, chat_message
):
    agent._last_confidence = 0.42
    ctx = chat_context()
    await agent.on_user_turn_completed(ctx, chat_message("I need help with my order"))

    assert len(ctx.added) == 1
    msg = ctx.added[0]
    assert msg["role"] == "system"
    assert msg["content"] == c.format_context("I need help with my order", 0.42)


async def test_turn_no_confidence_yet(agent, chat_context, chat_message):
    assert agent._last_confidence is None
    ctx = chat_context()
    await agent.on_user_turn_completed(ctx, chat_message("hello"))
    assert ctx.added == []


async def test_turn_injects_band_not_number(agent, chat_context, chat_message):
    for value, band in ((0.35, "low"), (0.92, "high")):
        agent._last_confidence = value
        ctx = chat_context()
        await agent.on_user_turn_completed(ctx, chat_message("do the thing"))
        content = ctx.added[0]["content"]
        assert f"STT confidence: {band}" in content
        assert f"{value:.2f}" not in content


async def test_turn_logs_numeric_confidence(agent, chat_context, chat_message, caplog):
    agent._last_confidence = 0.42
    with caplog.at_level("INFO", logger="confidence-agent"):
        await agent.on_user_turn_completed(chat_context(), chat_message("hi"))
    assert "confidence=0.42" in caplog.text
    assert "band=low" in caplog.text
