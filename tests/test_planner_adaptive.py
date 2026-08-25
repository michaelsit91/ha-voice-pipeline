import pytest
from unittest.mock import AsyncMock, MagicMock
from pipeline.agents import planner


_EMPTY = '{"corrected":"x","intent":"action","steps":[],"ok_response":"","already_response":"","fail_response":""}'
_GOOD = ('{"corrected":"x","intent":"action","steps":'
         '[{"domain":"light","service":"turn_on","entity_id":"light.a"}],'
         '"ok_response":"ok","already_response":"","fail_response":""}')


@pytest.mark.asyncio
async def test_retries_with_thinking_when_first_attempt_has_no_steps():
    ollama = MagicMock()
    ollama.chat = AsyncMock(side_effect=[_EMPTY, _GOOD])
    entities = [{"entity_id": "light.a", "name": "A", "state": "on"}]
    result = await planner.plan("turn on the light", entities, [], ollama)
    assert result["steps"], "should recover steps on the thinking retry"
    assert ollama.chat.await_count == 2
    # fast path first (no thinking), thinking only on retry
    assert ollama.chat.await_args_list[0].kwargs.get("think") in (False, None)
    assert ollama.chat.await_args_list[1].kwargs.get("think") is True


@pytest.mark.asyncio
async def test_no_retry_when_first_attempt_succeeds():
    ollama = MagicMock()
    ollama.chat = AsyncMock(side_effect=[_GOOD])
    entities = [{"entity_id": "light.a", "name": "A", "state": "on"}]
    result = await planner.plan("turn on the light", entities, [], ollama)
    assert result["steps"]
    assert ollama.chat.await_count == 1  # fast path only, no thinking cost
    assert ollama.chat.await_args_list[0].kwargs.get("think") in (False, None)
