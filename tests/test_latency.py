"""
Latency regression suite — tests the pipeline container DIRECTLY.

Direct testing bypasses HA routing overhead (~4.5s per call due to HA's conversation
agent infrastructure) and measures pure pipeline performance.

Run:
    export $(cat config.env | xargs)
    pytest tests/test_latency.py -v -s

Thresholds are stored in baseline.json. Update them after any intentional architectural
change by running: python -m pytest tests/test_latency.py -v -s --update-baseline
"""
import json, os, time, urllib.request, pytest
from pathlib import Path

CONTAINER_URL = os.getenv("PIPELINE_URL", "http://192.168.68.250:18795")
RUNS = 3

COMMANDS = {
    "status_query":  "is the living room light on",
    "single_action": "turn on the kitchen light",
    "multi_device":  "turn off all living room lights",
}

def _call_pipeline(text: str) -> tuple[float, str]:
    payload = json.dumps({"messages": [{"role": "user", "content": text}]}).encode()
    req = urllib.request.Request(
        f"{CONTAINER_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    text_out = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    return elapsed, text_out

def _load_baseline() -> dict:
    return json.loads((Path(__file__).parent / "baseline.json").read_text())

@pytest.fixture(scope="session", autouse=True)
def warmup_pipeline():
    """Send a throwaway request to ensure GPU is at full speed before timing."""
    print("\n  [warmup] pre-warming pipeline GPU...")
    try:
        _call_pipeline("hello")
        _call_pipeline("hello")  # Two calls to stabilize GPU P-state
    except Exception as e:
        print(f"  [warmup] warning: {e}")

@pytest.mark.parametrize("cmd_key,text", COMMANDS.items())
def test_latency_within_threshold(cmd_key, text):
    """Each command must complete within the fixed threshold from baseline.json."""
    baseline   = _load_baseline()
    threshold  = baseline["regression_thresholds_seconds"][cmd_key]
    times, last = [], ""
    for _ in range(RUNS):
        elapsed, resp = _call_pipeline(text)
        assert resp.strip(), f"Empty response for: {text}"
        times.append(elapsed)
        last = resp
    avg = sum(times) / len(times)
    print(f"\n  {cmd_key}: avg={avg:.2f}s  threshold={threshold:.1f}s  "
          f"times={[f'{t:.2f}' for t in times]}")
    print(f"  last response: {last[:80]}")
    assert avg <= threshold, (
        f"{cmd_key} avg {avg:.2f}s exceeds threshold {threshold:.1f}s. "
        f"If this is intentional, update baseline.json regression_thresholds_seconds."
    )

@pytest.mark.parametrize("cmd_key,text", COMMANDS.items())
def test_response_non_empty(cmd_key, text):
    _, resp = _call_pipeline(text)
    assert resp.strip(), f"Empty response for: {text}"

def test_health_endpoint():
    req = urllib.request.Request(f"{CONTAINER_URL}/health")
    with urllib.request.urlopen(req, timeout=5) as r:
        body = json.loads(r.read())
    assert body.get("status") == "ok"

def test_status_query_describes_state():
    _, resp = _call_pipeline(COMMANDS["status_query"])
    assert any(w in resp.lower() for w in ("on", "off", "those", "is", "all")), \
        f"Response doesn't describe state: {resp}"

def test_models_endpoint():
    req = urllib.request.Request(f"{CONTAINER_URL}/v1/models")
    with urllib.request.urlopen(req, timeout=5) as r:
        body = json.loads(r.read())
    assert body.get("object") == "list"
    assert len(body.get("data", [])) >= 1
