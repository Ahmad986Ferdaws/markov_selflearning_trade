"""Phase 2 comparison CLI."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from app.config import get_settings
from app.db import get_session_factory, init_db
from app.services.comparison import format_report, run_comparison


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="compare-cli",
        description="Compare baseline vs agent on run snapshots",
        allow_abbrev=False,
    )
    parser.add_argument("run_id", type=int, help="Run ID with stored snapshots")
    args = parser.parse_args(argv)
    init_db()
    settings = get_settings()
    db = get_session_factory()()
    try:
        try:
            report = run_comparison(db, args.run_id, settings)
        except ValueError as exc:
            # an unknown run id is a usage error (exit 2, message on stderr),
            # not a traceback
            parser.error(str(exc))
        print(format_report(report))
    finally:
        db.close()


if __name__ == "__main__":
    main()
