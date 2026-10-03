"""The experiment record: written by the commands, read back by anything, refused when it
is not one this xaig knows.

The numbers inside are not re-derived here (``test_latent_steering.py`` and
``test_latent_evaluate.py`` plant and check them); what is asserted is that a record holds
those numbers, who made them and how to make them again, in a form that is validated
on the way in.
"""

from __future__ import annotations

import json
import shlex

import pytest

np = pytest.importorskip("numpy")

from click.testing import CliRunner  # noqa: E402

from conftest import URL, write_evaluation_record, write_steering_record  # noqa: E402
from xaig._cli import cli  # noqa: E402
from xaig.core.errors import RequestError  # noqa: E402
from xaig.latents import (  # noqa: E402
    ExperimentRecord,
    Split,
    evaluation_record,
    load_record,
    open_source,
    record_from_dict,
    save_record,
)
from xaig.latents.record import SUPPORTED, VERSION  # noqa: E402


@pytest.fixture
def steering(tmp_path):
    return write_steering_record(tmp_path)


def test_the_steer_command_writes_a_record_of_what_it_found(steering):
    full, path = steering
    record = load_record(path)
    assert isinstance(record, ExperimentRecord)
    assert (record.kind, record.version) == ("steering", VERSION)
    assert json.loads(path.read_text())["format"] == "xaig.experiment-record"

    temperature = record.results["effects"]["temperature"]
    assert temperature["feature_response"] == full["effects"]["temperature"]["feature_response"]
    assert temperature["rank"] == 1 and len(temperature["random_responses"]) == 12
    assert record.results["spec"]["steer"]["feature"] == 0

    # Identity: which system, with which options, and which basis by content.
    assert record.provenance["model"] == "toy-dynamics"
    assert record.provenance["options"]["masked"] == 3
    assert record.provenance["xaig"]
    assert record.bases == (full["provenance"]["basis"],)
    assert len(record.bases[0]["sha256"]) == 64 and "basis" not in record.provenance
    assert record.split is None and record.commit is None


def test_a_record_is_compact_beside_the_result_it_summarises(steering):
    """The runs and pairings are the bulk of a result, and a record leaves them out."""
    full, path = steering
    assert {"runs", "pairings"} <= full.keys()
    record = json.loads(path.read_text())
    assert not {"runs", "pairings"} & record["results"].keys()
    assert path.stat().st_size < len(json.dumps(full)) / 2


def test_the_command_in_a_steering_record_reproduces_it(steering):
    _, path = steering
    record = load_record(path)
    argv = shlex.split(record.command)
    assert argv[:3] == ["xaig", "latents", "steer"]
    assert f"--basis-sha256 {record.bases[0]['sha256']}" in record.command
    assert "--record" not in argv  # the command redoes the work, not the writing of this file
    again = CliRunner().invoke(cli, [*argv[1:], "--json"])
    assert again.exit_code == 0, again.output
    assert json.loads(again.output)["effects"] == record.results["effects"]


@pytest.fixture
def evaluation(tmp_path):
    full, path, fits = write_evaluation_record(tmp_path)
    return full, load_record(path), fits


def test_the_evaluate_command_writes_the_split_and_the_numbers(evaluation):
    full, record, fits = evaluation
    assert record.kind == "evaluation"
    # The split is the one the command computed, as time labels, and held once.
    assert record.split == full["split"] and len(record.split["test"]) == 2
    assert all("split" not in e for e in record.results["evaluations"])
    first = record.results["evaluations"][0]
    assert first["test"] == full["evaluations"][0]["test"] and first["fitted_on"] == "train"
    assert first["basis"] == full["evaluations"][0]["provenance"]["basis"]
    assert [p["label"] for p in record.results["curve"]["points"]] == ["pca-1", "pca-4"]
    assert record.results["stability"]["n_dictionaries"] == 2
    assert [b["path"] for b in record.bases] == fits
    assert record.settings["mask_variable"] == "sst" and record.settings["pca"] == [1, 4]
    assert record.provenance["options"] == {"mask_variable": "sst"}


def test_the_command_in_an_evaluation_record_reproduces_it(evaluation):
    _, record, _ = evaluation
    argv = shlex.split(record.command)
    assert argv[:3] == ["xaig", "latents", "evaluate"] and "--split-only" not in argv
    again = CliRunner().invoke(cli, [*argv[1:], "--json"])
    assert again.exit_code == 0, again.output
    redone = json.loads(again.output)
    assert redone["split"] == record.split
    assert [e["test"] for e in redone["evaluations"]] == [
        e["test"] for e in record.results["evaluations"]
    ]
    assert redone["curve"]["points"] == record.results["curve"]["points"]


def test_the_command_in_an_evaluation_record_pins_each_basis_by_content(evaluation):
    _, record, fits = evaluation
    argv = shlex.split(record.command)
    pairs = [
        (argv[i + 1], argv[i + 3]) for i, w in enumerate(argv) if w == "--basis"
    ]  # --basis PATH --basis-sha256 HASH, in order
    assert [p for p, _ in pairs] == fits
    assert [h for _, h in pairs] == [b["sha256"] for b in record.bases]
    assert all(argv[argv.index(p) + 1] == "--basis-sha256" for p, _ in pairs)
    # A basis file changed since is refused by the very command the record holds.
    refit = ["latents", "pca", argv[3], "--components", "3", "--out", fits[0], "--time", "0"]
    assert CliRunner().invoke(cli, refit).exit_code == 0
    again = CliRunner().invoke(cli, [*argv[1:], "--json"])
    assert again.exit_code != 0 and "sha256" in again.output


def test_a_record_cannot_be_asked_of_a_split_alone(tmp_path):
    archive = str(tmp_path / "toy")
    assert CliRunner().invoke(cli, ["latents", "toy", archive]).exit_code == 0
    done = CliRunner().invoke(
        cli, ["latents", "evaluate", archive, "--split-only", "--record", str(tmp_path / "r.json")]
    )
    assert done.exit_code != 0 and "--split-only" in done.output
    assert not (tmp_path / "r.json").exists()


def test_a_record_names_the_commit_a_hub_archive_was_opened_at(hub):
    info = open_source(URL, revision="v1").info()
    split = Split(scheme="blocks", train=(info.times[0],), test=(info.times[1],))
    record = evaluation_record(info, split, settings={}, command="xaig latents evaluate ...")
    assert record.commit == "def"
    assert record.provenance["revision"] == {"requested": "v1", "commit": "def"}


# -- reading: what is refused ----------------------------------------------------------


@pytest.fixture
def good(steering):
    return json.loads(steering[1].read_text())


def test_a_record_survives_a_trip_through_its_file(steering, tmp_path):
    _, path = steering
    again = save_record(tmp_path / "copy.json", load_record(path))
    assert json.loads(again.read_text()) == json.loads(path.read_text())


@pytest.mark.parametrize("version", [0, 2, 99, "1", 1.0, 1.5, None, True])
def test_a_version_this_xaig_does_not_read_is_refused(good, version):
    assert SUPPORTED == (1,)
    with pytest.raises(RequestError, match="schema version"):
        record_from_dict({**good, "schema_version": version})


def test_a_newer_record_says_to_upgrade(good):
    with pytest.raises(RequestError, match="newer xaig"):
        record_from_dict({**good, "schema_version": VERSION + 1})


def test_what_is_not_a_record_is_refused_in_one_line(good, tmp_path):
    with pytest.raises(RequestError, match="not an xaig experiment record"):
        record_from_dict({**good, "format": "something-else"})
    with pytest.raises(RequestError, match="must be a mapping"):
        record_from_dict([1, 2])
    with pytest.raises(RequestError, match="kind 'survey'"):
        record_from_dict({**good, "kind": "survey"})
    plain = tmp_path / "plain.txt"
    plain.write_text("not json {")
    with pytest.raises(RequestError, match="not JSON"):
        load_record(plain)
    with pytest.raises(RequestError, match="cannot read"):
        load_record(tmp_path / "absent.json")


@pytest.mark.parametrize("missing", ["provenance", "settings", "results", "command", "bases"])
def test_a_record_with_a_part_missing_is_refused(good, missing):
    del good[missing]
    with pytest.raises(RequestError, match=missing):
        record_from_dict(good)


def test_what_a_later_writer_adds_within_a_version_is_ignored(good):
    record = record_from_dict({**good, "added_later": {"anything": 1}})
    assert record.kind == "steering" and "added_later" not in record.to_dict()


def _set(data, value, *path):
    out = json.loads(json.dumps(data))
    node = out
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return out


STEERING_BREAKS = [
    (None, "results", "effects", "temperature", "feature_response"),
    ("2", "results", "spec", "steer", "amount"),
    (float("nan"), "results", "effects", "temperature", "reconstruction_response"),
    ({}, "bases"),
]


@pytest.mark.parametrize("change", STEERING_BREAKS, ids=lambda c: ".".join(c[1:]) + f"={c[0]!r}")
def test_a_steering_record_of_the_wrong_types_is_a_request_error(good, change):
    value, *path = change
    with pytest.raises(RequestError):
        record_from_dict(_set(good, value, *path))


EVALUATION_BREAKS = [
    ({"a": 1}, "results", "evaluations"),
    (None, "split"),
    ("x", "results", "evaluations", 0, "test", "explained_variance"),
    (True, "results", "stability", "threshold"),
]


@pytest.mark.parametrize("change", EVALUATION_BREAKS, ids=lambda c: ".".join(map(str, c[1:])))
def test_an_evaluation_record_of_the_wrong_types_is_a_request_error(evaluation, change):
    _, record, _ = evaluation
    value, *path = change
    with pytest.raises(RequestError):
        record_from_dict(_set(record.to_dict(), value, *path))


def test_a_number_that_was_not_one_is_null_and_still_a_record(evaluation):
    _, record, _ = evaluation
    data = _set(record.to_dict(), None, "results", "evaluations", 0, "test", "explained_variance")
    data = _set(data, None, "results", "curve", "points", 0, "pca_at_same_sparsity")
    assert record_from_dict(data).kind == "evaluation"
