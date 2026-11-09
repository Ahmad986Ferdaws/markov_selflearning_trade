"""compare-cli argument handling (app/cli/compare.py).

init_db is replaced by a no-op and the session factory by one bound to an
in-memory SQLite database with the tables created, so nothing touches
paper_trading.db. An unknown run id used to escape as a ValueError
traceback; it is now an argparse usage error on stderr with exit code 2.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.cli import compare as cli
from app.models.base import Base


@pytest.fixture
def empty_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(cli, "init_db", lambda: None)
    monkeypatch.setattr(cli, "get_session_factory", lambda: sessionmaker(bind=engine))


def test_an_unknown_run_id_is_a_usage_error_not_a_traceback(empty_db, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["999"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "Run 999 not found" in captured.err
    assert captured.out == ""                       # no report was printed
    assert captured.err.startswith("usage: compare-cli")


def test_a_non_integer_run_id_is_also_a_usage_error(empty_db, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["latest"])
    assert exc.value.code == 2
    assert "invalid int value" in capsys.readouterr().err


def test_help_names_the_program_and_the_run_id_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: compare-cli")
    assert "run_id" in out
