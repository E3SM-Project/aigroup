from __future__ import annotations

import pytest

from xaig.core.model import Campaign, MetricSeries, Run, RunStatus


def test_interrupted_is_not_failed():
    """A preempted-and-requeued job is the normal state of a live campaign."""
    assert RunStatus.INTERRUPTED is not RunStatus.FAILED
    assert str(RunStatus.INTERRUPTED) == "interrupted"


def test_metric_series_rejects_ragged_input():
    with pytest.raises(ValueError, match="2 steps vs 1 values"):
        MetricSeries("loss", steps=(1.0, 2.0), values=(0.5,))


def test_metric_at_does_not_interpolate():
    """A missing epoch must read as missing, never as a neighbour's number."""
    series = MetricSeries("loss", steps=(1.0, 3.0), values=(0.9, 0.4))
    assert series.at(1.0) == 0.9
    assert series.at(2.0) is None
    assert series.last == 0.4


def test_empty_metric_has_no_last():
    assert MetricSeries("loss").last is None


def test_attrs_compare_as_strings():
    """So `--select batch=16` from the CLI matches an int attribute."""
    run = Run("a", {"batch": 16})
    assert run.matches(batch="16")
    assert run.matches(batch=16)
    assert not run.matches(batch=8)


def test_with_attrs_does_not_mutate():
    run = Run("a", {"x": 1})
    assert run.with_attrs(y=2).attrs == {"x": 1, "y": 2}
    assert run.attrs == {"x": 1}


@pytest.fixture
def campaign():
    return Campaign(
        "c",
        (
            Run("b", {"realm": "ocn", "seed": 2}),
            Run("a", {"realm": "atm", "seed": 1}),
            Run("c", {"realm": "ocn", "seed": 1}),
        ),
    )


def test_filter_and_get(campaign):
    assert len(campaign.filter(realm="ocn")) == 2
    assert campaign.get("a").attrs["realm"] == "atm"
    assert campaign.get("missing") is None


def test_sorted_by(campaign):
    assert [r.id for r in campaign.sorted_by("realm", "seed")] == ["a", "c", "b"]


def test_attr_names_is_union_in_first_seen_order():
    c = Campaign("c", (Run("a", {"x": 1}), Run("b", {"y": 2, "x": 3})))
    assert c.attr_names() == ["x", "y"]


def test_table_includes_id_and_status(campaign):
    row = campaign.table(["realm"])[0]
    assert set(row) == {"id", "status", "realm"}
    assert row["status"] == "unknown"
