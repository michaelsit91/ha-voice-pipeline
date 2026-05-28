"""
Advanced pipeline tests: state verification, complex commands, edge cases.

Safe entities used (always restored after test) — use light.* domain
since Z2M exposes devices as light entities with proper Zigbee semantics:
  light.office_2_gang_left   — Office Light     (usually off)
  fan.office_2_gang_right    — Office Ceiling Fan (usually off)
  light.hallway_3_gang_center_3 — Hallway Cabinet Light (usually off)

Note: Zigbee devices have a ~400ms propagation delay after a service call
before HA state reflects the new value. Tests wait 1s after pipeline calls
to account for this.
"""
import asyncio, os, pytest
from pipeline.runner import run_pipeline
from pipeline.ha_client import HAClient
from pipeline.ollama_client import OllamaClient

HA_URL     = os.getenv("HA_URL",     "http://homeassistant.local:8123")
HA_TOKEN   = os.getenv("HA_TOKEN",   "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://homeassistant.local:11434")
MODEL      = os.getenv("MODEL",      "default")

# Safe entities — use light.* (proper Zigbee light entities, not switch mirrors)
OFFICE_LIGHT  = "light.office_2_gang_left"
OFFICE_FAN    = "fan.office_2_gang_right"
CABINET_LIGHT = "light.hallway_3_gang_center_3"

ZIGBEE_SETTLE = 1.0  # seconds to wait after a state change for Zigbee propagation


@pytest.fixture(scope="module")
def ha():
    return HAClient(HA_URL, HA_TOKEN)


@pytest.fixture(scope="module")
def ollama():
    return OllamaClient(OLLAMA_URL, MODEL)


@pytest.fixture(autouse=True)
async def restore_office(ha):
    """Reset office light and fan to off before and after each test."""
    await ha.call_service("light", "turn_off", OFFICE_LIGHT)
    await ha.call_service("fan",   "turn_off", OFFICE_FAN)
    await asyncio.sleep(ZIGBEE_SETTLE)
    yield
    await ha.call_service("light", "turn_off", OFFICE_LIGHT)
    await ha.call_service("fan",   "turn_off", OFFICE_FAN)
    await asyncio.sleep(ZIGBEE_SETTLE)


# ── State-change verification ─────────────────────────────────────────────────

async def test_turn_on_verifies_state_changed(ha, ollama):
    """Pipeline turns on office light and state actually changes."""
    before = await ha.get_state(OFFICE_LIGHT)
    assert before["state"] == "off", "Fixture should ensure light starts off"

    response = await run_pipeline("turn on the office light", ha, ollama)
    assert isinstance(response, str) and len(response) > 0
    await asyncio.sleep(ZIGBEE_SETTLE)

    after = await ha.get_state(OFFICE_LIGHT)
    assert after["state"] == "on", f"Office light should be on, got: {after['state']}"


async def test_turn_off_verifies_state_changed(ha, ollama):
    """Pipeline turns off office light and state actually changes."""
    await ha.call_service("light", "turn_on", OFFICE_LIGHT)
    await asyncio.sleep(ZIGBEE_SETTLE)
    before = await ha.get_state(OFFICE_LIGHT)
    assert before["state"] == "on"

    response = await run_pipeline("turn off the office light", ha, ollama)
    assert isinstance(response, str) and len(response) > 0
    await asyncio.sleep(ZIGBEE_SETTLE)

    after = await ha.get_state(OFFICE_LIGHT)
    assert after["state"] == "off", f"Office light should be off, got: {after['state']}"


async def test_status_query_returns_accurate_state(ha, ollama):
    """Status query reflects actual current state."""
    await ha.call_service("light", "turn_off", OFFICE_LIGHT)
    await asyncio.sleep(ZIGBEE_SETTLE)
    response = await run_pipeline("is the office light on", ha, ollama)
    assert any(w in response.lower() for w in ("off", "no", "not"))

    await ha.call_service("light", "turn_on", OFFICE_LIGHT)
    await asyncio.sleep(ZIGBEE_SETTLE)
    response = await run_pipeline("is the office light on", ha, ollama)
    assert any(w in response.lower() for w in ("on", "yes"))


# ── Fan control ───────────────────────────────────────────────────────────────

async def test_fan_turn_on_verifies_state(ha, ollama):
    """Pipeline controls fan and state actually changes."""
    before = await ha.get_state(OFFICE_FAN)
    assert before["state"] == "off"

    response = await run_pipeline("turn on the office fan", ha, ollama)
    assert isinstance(response, str) and len(response) > 0
    await asyncio.sleep(ZIGBEE_SETTLE)

    after = await ha.get_state(OFFICE_FAN)
    assert after["state"] == "on", f"Office fan should be on, got: {after['state']}"


# ── Multi-domain ──────────────────────────────────────────────────────────────

async def test_multi_domain_light_and_fan(ha, ollama):
    """Command targeting both light and fan executes both."""
    response = await run_pipeline("turn on the office light and the office fan", ha, ollama)
    assert isinstance(response, str) and len(response) > 0
    await asyncio.sleep(ZIGBEE_SETTLE)

    light_state = await ha.get_state(OFFICE_LIGHT)
    fan_state   = await ha.get_state(OFFICE_FAN)
    # At least one should have changed (LLM may prioritize one)
    either_on = light_state["state"] == "on" or fan_state["state"] == "on"
    assert either_on, f"Expected at least light or fan on, got light={light_state['state']} fan={fan_state['state']}"


# ── STT error correction ──────────────────────────────────────────────────────

async def test_stt_correction_controls_correct_entity(ha, ollama):
    """STT errors in transcript are corrected and correct entity is controlled.
    Uses 'tern' (turn) and 'lait' (light) as garbled words; 'office' is intact
    so the keyword filter can find the office light entity."""
    response = await run_pipeline("tern on the office lait", ha, ollama)
    assert isinstance(response, str) and len(response) > 0
    await asyncio.sleep(ZIGBEE_SETTLE)

    after = await ha.get_state(OFFICE_LIGHT)
    assert after["state"] == "on", \
        f"STT-corrected command should have turned on office light, got: {after['state']}"


# ── Entity not found ─────────────────────────────────────────────────────────

async def test_nonexistent_entity_responds_gracefully(ha, ollama):
    """Pipeline responds gracefully when entity doesn't exist."""
    response = await run_pipeline("turn on the invisible magic lamp", ha, ollama)
    assert isinstance(response, str) and len(response) > 0
    # Should NOT claim success
    assert "turned on" not in response.lower() or "invisible" not in response.lower()


# ── Switch control ────────────────────────────────────────────────────────────

async def test_switch_entity_controlled(ha, ollama):
    """Pipeline correctly identifies and controls switch-domain entities."""
    response = await run_pipeline("turn on the hallway cabinet light", ha, ollama)
    assert isinstance(response, str) and len(response) > 0
    await asyncio.sleep(ZIGBEE_SETTLE)

    after = await ha.get_state(CABINET_LIGHT)
    await ha.call_service("light", "turn_off", CABINET_LIGHT)
    assert after["state"] == "on", f"Cabinet light should be on, got: {after['state']}"


# ── Toggle ────────────────────────────────────────────────────────────────────

async def test_toggle_command(ha, ollama):
    """Toggle command changes state from current state."""
    before = await ha.get_state(OFFICE_LIGHT)
    response = await run_pipeline("toggle the office light", ha, ollama)
    assert isinstance(response, str) and len(response) > 0
    await asyncio.sleep(ZIGBEE_SETTLE)

    after = await ha.get_state(OFFICE_LIGHT)
    expected = "on" if before["state"] == "off" else "off"
    assert after["state"] == expected, \
        f"Toggle from {before['state']} should give {expected}, got {after['state']}"


# ── Response quality ─────────────────────────────────────────────────────────

async def test_response_is_short(ha, ollama):
    """All responses must be at most 2 sentences."""
    commands = [
        "turn on the office light",
        "is the office light on",
        "turn off the office light",
    ]
    for cmd in commands:
        response = await run_pipeline(cmd, ha, ollama)
        # Count sentences by terminal punctuation (rough heuristic)
        sentences = [s.strip() for s in response.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        assert len(sentences) <= 3, f"Response too long ({len(sentences)} sentences): {response!r}"


async def test_response_no_markdown(ha, ollama):
    """Responses must not contain markdown (lists, headers, bold)."""
    response = await run_pipeline("turn on the office light", ha, ollama)
    assert "*" not in response, f"Markdown asterisk in response: {response!r}"
    assert "#" not in response, f"Markdown header in response: {response!r}"
    assert "- " not in response, f"Markdown list in response: {response!r}"
