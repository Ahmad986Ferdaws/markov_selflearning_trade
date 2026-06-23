"""Phase 2 comparison CLI."""

import argparse

from app.config import get_settings
from app.db import get_session_factory, init_db
from app.services.comparison import format_report, run_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs agent on run snapshots")
    parser.add_argument("run_id", type=int, help="Run ID with stored snapshots")
    args = parser.parse_args()
    init_db()
    settings = get_settings()
    db = get_session_factory()()
    try:
        report = run_comparison(db, args.run_id, settings)
        print(format_report(report))
    finally:
        db.close()


if __name__ == "__main__":
    main()
