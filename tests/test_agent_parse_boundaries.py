import pytest

from app.services.agent_policy import _parse_position


@pytest.mark.parametrize('raw', ['{"position":NaN}', '{"position":Infinity}',
    '{"position":true}', '{"position":null,"reasoning":"0.8"}',
    '{"position":"bad","reasoning":"0.8"}', 'year 2026; confidence 0.5',
    'position: 50%', 'position: 0.5oops'])
def test_invalid_or_ambiguous_reply_holds(raw):
    assert _parse_position(raw, .3) == (.3, 'unparseable_hold')


@pytest.mark.parametrize('raw,expected', [('position: 1e-2', .01),
    ('position: .25', .25), ('1e-2', .01), ('{"position":4.2}', 1.),
    ('exposure to 0.8', .8)])
def test_complete_numbers_and_finite_clamping(raw, expected):
    assert _parse_position(raw, .3)[0] == expected


@pytest.mark.parametrize('raw', ['position: +0.5+0.2', 'position: .5/.2',
    'position: 0.5-0.7', 'position: .5 + .2', 'position: .5 / .2',
    '[' * 2000 + '0.5' + ']' * 2000])
def test_malformed_expressions_and_nested_json_hold(raw):
    assert _parse_position(raw, .3) == (.3, 'unparseable_hold')
