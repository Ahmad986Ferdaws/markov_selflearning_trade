"""The run_id foreign keys the legacy queries filter on carry an index.

Every /runs query filters snapshots, trades and metrics by run_id, and
SQLite does not index a foreign key on its own. create_all on an in-memory
engine must therefore produce a single-column run_id index on each table.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect

from app.models.base import Base


@pytest.fixture(scope="module")
def inspector():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return inspect(engine)


@pytest.mark.parametrize("table", ["snapshots", "trades", "metrics"])
def test_each_child_table_has_an_index_on_run_id_alone(inspector, table):
    indexes = inspector.get_indexes(table)
    run_id_indexes = [ix for ix in indexes if ix["column_names"] == ["run_id"]]
    assert len(run_id_indexes) == 1, indexes
    assert not run_id_indexes[0]["unique"]          # many rows per run


def test_the_parent_runs_table_gained_no_index():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    assert inspect(engine).get_indexes("runs") == []
