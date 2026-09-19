from __future__ import annotations

import pytest
import yaml

from conftest import TOY_SPEC
from xaig.adapters.table import TableDiscoverer
from xaig.caig import spec as spec_module
from xaig.core.errors import AdapterError
from xaig.core.model import RunStatus


@pytest.fixture
def toy_spec():
    return spec_module.from_mapping(yaml.safe_load(TOY_SPEC))


def test_discovers_every_row(toy_table_path, toy_spec):
    runs = list(TableDiscoverer(toy_table_path, id_column="name", spec=toy_spec).discover())
    assert [r.id for r in runs] == [
        "run001-fast-d1-w08-s1",
        "run002-fast-d2-w08-s1",
        "run003-slow-d1-w16-s2",
    ]


def test_merges_columns_and_parsed_attributes(toy_table_path, toy_spec):
    run = next(TableDiscoverer(toy_table_path, id_column="name", spec=toy_spec).discover())
    assert run.attrs["note"] == "baseline"  # from the column
    assert run.attrs["depth"] == 1  # from the id


def test_spec_attributes_win_over_columns(tmp_path, toy_spec):
    """A `mode` column disagreeing with the id must not shadow the parsed value:
    the id is the primary key and the spec gives its values a known type."""
    path = tmp_path / "t.csv"
    path.write_text("name,mode\nrun001-fast-d1-w08-s1,WRONG\n")
    run = next(TableDiscoverer(path, id_column="name", spec=toy_spec).discover())
    assert run.attrs["mode"] == "fast"


def test_a_conflicting_column_loses_but_is_not_forgotten(tmp_path, toy_spec):
    """The id wins, and the disagreement stays on the run for `check` to report."""
    path = tmp_path / "t.csv"
    path.write_text("name,seed\nrun001-fast-d1-w08-s1,99\n")
    run = next(TableDiscoverer(path, id_column="name", spec=toy_spec).discover())
    assert run.attrs["seed"] == 1
    assert [i.kind for i in run.issues] == ["metadata"]
    assert "seed=99" in run.issues[0].message


@pytest.mark.parametrize("column", ["1", "01", "s1", "S01"])
def test_a_column_repeating_the_ids_notation_is_not_a_conflict(tmp_path, toy_spec, column):
    path = tmp_path / "t.csv"
    path.write_text(f"name,seed\nrun001-fast-d1-w08-s1,{column}\n")
    run = next(TableDiscoverer(path, id_column="name", spec=toy_spec).discover())
    assert run.attrs["seed"] == 1 and run.issues == ()


def test_columns_cannot_shadow_the_runs_own_id_and_status(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("name,id,status\nr01,shadow,finished\n")
    run = next(TableDiscoverer(path, id_column="name").discover())
    assert (run.id, run.status) == ("r01", RunStatus.UNKNOWN)
    assert run.attrs == {"table_id": "shadow", "table_status": "finished"}


def test_status_is_read_only_from_the_column_named_for_it(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("name,state\na,FINISHED\nb,done\nc,exploded\nd,\n")
    runs = list(
        TableDiscoverer(
            path, id_column="name", status_column="state", status_map={"done": "finished"}
        ).discover()
    )
    assert [r.status for r in runs] == [
        RunStatus.FINISHED,
        RunStatus.FINISHED,
        RunStatus.UNKNOWN,
        RunStatus.UNKNOWN,
    ]
    assert [i.kind for i in runs[2].issues] == ["status"]
    assert all("state" not in r.attrs for r in runs)


def test_status_map_must_map_onto_real_statuses(tmp_path):
    with pytest.raises(AdapterError, match="statuses are"):
        TableDiscoverer(tmp_path / "t.csv", status_map={"done": "completed"})


def test_delimiter_is_sniffed(tmp_path, toy_spec):
    path = tmp_path / "t.tsv"
    path.write_text("name\tnote\nrun001-fast-d1-w08-s1\thi\n")
    runs = list(TableDiscoverer(path, id_column="name", spec=toy_spec).discover())
    assert runs[0].attrs["note"] == "hi"


def test_blank_rows_are_skipped(tmp_path, toy_spec):
    path = tmp_path / "t.csv"
    path.write_text("name,note\nrun001-fast-d1-w08-s1,a\n,\n")
    assert len(list(TableDiscoverer(path, id_column="name", spec=toy_spec).discover())) == 1


def test_unparseable_id_is_tolerated_by_default(tmp_path, toy_spec):
    """A partially-readable campaign is more useful than an exception."""
    path = tmp_path / "t.csv"
    path.write_text("name,note\nnot-a-valid-id,x\n")
    run = next(TableDiscoverer(path, id_column="name", spec=toy_spec).discover())
    assert run.id == "not-a-valid-id"
    assert run.attrs["note"] == "x"
    assert [i.kind for i in run.issues] == ["id"]


def test_strict_mode_rejects_an_unparseable_id(tmp_path, toy_spec):
    path = tmp_path / "t.csv"
    path.write_text("name,note\nnot-a-valid-id,x\n")
    with pytest.raises(Exception, match="does not match"):
        list(TableDiscoverer(path, id_column="name", spec=toy_spec, strict=True).discover())


def test_missing_file_is_a_clear_error(tmp_path, toy_spec):
    with pytest.raises(AdapterError, match="table not found"):
        list(TableDiscoverer(tmp_path / "nope.csv", id_column="name").discover())


def test_missing_id_column_names_what_it_found(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("other,note\na,b\n")
    with pytest.raises(AdapterError, match="no 'name' column"):
        list(TableDiscoverer(path, id_column="name").discover())


def test_empty_file_yields_nothing(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("")
    assert list(TableDiscoverer(path, id_column="name").discover()) == []


def test_works_without_a_spec(toy_table_path):
    runs = list(TableDiscoverer(toy_table_path, id_column="name").discover())
    assert runs[0].attrs == {"mode": "fast", "note": "baseline"}


def test_a_spreadsheets_byte_order_mark_is_not_part_of_a_column_name(tmp_path):
    path = tmp_path / "t.csv"
    path.write_bytes(b"\xef\xbb\xbfname,note\nr1,a\n")
    run = next(TableDiscoverer(path, id_column="name").discover())
    assert (run.id, run.attrs) == ("r1", {"note": "a"})


def test_a_quoted_field_may_hold_a_line_break(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text('name,note\nr1,"two\nlines"\nr2,b\n')
    runs = list(TableDiscoverer(path, id_column="name").discover())
    assert [r.id for r in runs] == ["r1", "r2"] and runs[0].attrs["note"] == "two\nlines"
