import json
from dataclasses import dataclass, replace

import pytest

from app.services.data_cache import save_run_record


@dataclass
class Report:
    symbol: str = 'SYN'
    data_hash: str = 'abcdef12' * 8
    total_return: float = .1


def test_distinct_reports_on_same_data_survive(tmp_path):
    first = save_run_record(Report(), tmp_path)
    second = save_run_record(replace(Report(), total_return=.2), tmp_path)
    assert first != second
    assert json.loads(first.read_text())['total_return'] == .1
    assert json.loads(second.read_text())['total_return'] == .2
    assert save_run_record(Report(), tmp_path) == first
    assert len(list(tmp_path.iterdir())) == 2


def test_old_evidence_is_preserved(tmp_path):
    legacy = tmp_path / 'SYN_abcdef12.json'
    legacy.write_text('historical evidence')
    save_run_record(Report(), tmp_path)
    assert legacy.read_text() == 'historical evidence'


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_record_not_written(tmp_path, bad):
    with pytest.raises(ValueError):
        save_run_record(replace(Report(), total_return=bad), tmp_path)
    assert not list(tmp_path.iterdir())


def test_record_name_is_contained_and_deterministic(tmp_path):
    report = replace(Report(), symbol='../odd/symbol')
    path = save_run_record(report, tmp_path)
    assert path.parent == tmp_path
    assert path == save_run_record(report, tmp_path)


def test_replayed_report_becomes_latest_without_changing_bytes(tmp_path):
    import os
    from app.cli.diagnose import _latest_record
    first = save_run_record(Report(), tmp_path)
    second = save_run_record(replace(Report(), total_return=.2), tmp_path)
    os.utime(first, (100, 100)); os.utime(second, (200, 200))
    content = first.read_bytes()
    save_run_record(Report(), tmp_path)
    assert _latest_record(tmp_path) == first
    assert first.read_bytes() == content
