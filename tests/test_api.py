from __future__ import annotations

import pytest

from xaig.caig import check_campaign, load_campaign, load_spec
from xaig.caig import spec as spec_module
from xaig.core import registry
from xaig.core.errors import AdapterError
from xaig.core.model import Campaign, MetricSeries, Run, RunStatus


class Everything:
    """One object, three protocols -- how a real framework adapter is shaped."""

    def __init__(self, source):
        self.source = source

    def discover(self):
        yield Run("a", location=self.source)
        yield Run("b", status=RunStatus.FAILED)

    def probe(self, run):
        return RunStatus.RUNNING

    def metrics(self, run, names=None):
        return {n: MetricSeries(n, (1.0,), (0.5,)) for n in names or ()}


@pytest.fixture
def everything():
    registry.register("everything", Everything)
    yield spec_module.from_mapping({"name": "t", "discovery": {"adapter": "everything"}})
    registry.unregister("everything")


def test_a_probe_settles_only_what_discovery_left_unknown(everything):
    campaign = load_campaign(everything, source="somewhere")
    assert [r.status for r in campaign] == [RunStatus.RUNNING, RunStatus.FAILED]
    assert campaign.get("a").location == "somewhere"


def test_metrics_are_attached_only_when_asked_for(everything):
    assert load_campaign(everything, source="s").get("a").metrics == {}
    series = load_campaign(everything, source="s", metrics=["loss"]).get("a").metrics["loss"]
    assert series.last == 0.5


def test_something_that_cannot_discover_is_not_a_campaign_adapter():
    registry.register("inert", lambda source=None: object())
    try:
        with pytest.raises(AdapterError, match="does not implement Discoverer"):
            load_campaign(spec_module.from_mapping({"name": "t"}), adapter="inert")
    finally:
        registry.unregister("inert")


def test_a_typo_in_the_discovery_block_is_an_error(toy_table_path):
    spec = spec_module.from_mapping(
        {"name": "t", "discovery": {"adapter": "table", "id_colum": "name"}}
    )
    with pytest.raises(AdapterError, match="id_colum"):
        load_campaign(spec, source=toy_table_path)


def test_the_discovery_block_may_carry_a_default_source(toy_table_path, tmp_path):
    spec = spec_module.from_mapping(
        {
            "name": "t",
            "discovery": {"adapter": "table", "id_column": "name", "source": str(toy_table_path)},
        }
    )
    assert len(load_campaign(spec)) == 3
    other = tmp_path / "other.csv"
    other.write_text("name\nonly-one\n")
    assert len(load_campaign(spec, source=other)) == 1


def test_overriding_the_adapter_drops_options_written_for_the_specs_own(toy_spec_path, everything):
    """toy's discovery block says id_column=name, which `everything` would reject."""
    campaign = load_campaign(load_spec(toy_spec_path), source="s", adapter="everything")
    assert len(campaign) == 2


def test_check_separates_id_problems_from_metadata_problems(toy_spec_path, tmp_path):
    table = tmp_path / "t.csv"
    table.write_text(
        "name,seed\n"
        "run001-fast-d1-w08-s1,1\n"  # fine
        "run002-fast-d2-w08-s1,99\n"  # column contradicts the id
        "run3-fast-d1-w08-s1,1\n"  # parses, but is not zero-padded: does not round-trip
        "garbage,1\n"  # does not parse at all
    )
    spec = load_spec(toy_spec_path)
    findings = check_campaign(load_campaign(spec, source=table), spec)
    assert sorted((f.run_id, f.kind) for f in findings) == [
        ("garbage", "id"),
        ("run002-fast-d2-w08-s1", "metadata"),
        ("run3-fast-d1-w08-s1", "id"),
    ]


def test_check_reports_a_duplicated_id_once():
    spec = spec_module.from_mapping({"name": "opaque"})
    findings = check_campaign(Campaign("c", (Run("x"), Run("x"), Run("y"))), spec)
    assert [(f.run_id, f.kind, f.message) for f in findings] == [("x", "id", "appears 2 times")]
