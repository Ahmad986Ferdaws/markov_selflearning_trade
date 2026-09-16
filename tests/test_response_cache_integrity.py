import json

import pytest

from app.services.agent_policy import ResponseCache


@pytest.mark.parametrize('payload', [None, [], ['x'], 1, {'x': 1}, {'x': None}])
def test_structurally_invalid_cache_recovers_as_empty(tmp_path, payload):
    path = tmp_path / 'cache.json'
    path.write_text(json.dumps(payload))
    cache = ResponseCache(path)
    assert not cache.contains('x')
    assert cache.get_or_call('x', lambda: 'reply') == 'reply'
    assert cache.misses == 1


def test_nontext_provider_result_is_not_cached():
    cache = ResponseCache()
    with pytest.raises(ValueError, match='text'):
        cache.get_or_call('x', lambda: {'position': .5})
    assert not cache.contains('x')
    assert cache.misses == 1


def test_failed_atomic_replace_preserves_previous_cache(tmp_path, monkeypatch):
    path = tmp_path / 'cache.json'
    path.write_text('{"old": "reply"}')
    cache = ResponseCache(path)
    cache.get_or_call('new', lambda: 'new reply')
    def fail(*args):
        raise OSError('simulated interrupted save')
    monkeypatch.setattr('os.replace', fail)
    with pytest.raises(OSError):
        cache.save()
    assert json.loads(path.read_text()) == {'old': 'reply'}
    assert list(tmp_path.iterdir()) == [path]


def test_valid_cache_roundtrip_never_calls_provider(tmp_path):
    path = tmp_path / 'cache.json'
    cache = ResponseCache(path)
    cache.get_or_call('x', lambda: 'reply')
    cache.save()
    def forbidden():
        pytest.fail('cache replay must be offline')
    assert ResponseCache(path).get_or_call('x', forbidden) == 'reply'
