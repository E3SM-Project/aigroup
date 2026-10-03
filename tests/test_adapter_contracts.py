"""What every adapter must do, asked of every adapter.

A new adapter is not finished when it reads its own files; it is finished when it passes
this module. Add it to ``LATENT_CASES`` (and to ``INTERVENABLE_CASES`` if it can be run) and
every test below is asked of it; ``test_every_shipped_adapter_is_under_contract`` fails
until you do. Nothing here names a layout: a case says how to make a source of its kind,
and the tests ask only what the protocols promise.

Two kinds of case. A *planted* one writes known arrays through the adapter's own ``write``
and is asked for them back, exactly (the answers are in this file, not read off the
adapter). A *found* one is a source made some other way -- the toy emulator's archive, the
hand-written one the other tests share -- and is asked only what holds of any source:
shapes, selection, copies, calendars, provenance, determinism, refusals.

The ``Intervenable`` contract is asked the same way of every system that can be run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("xarray")

from conftest import write_latent_archive  # noqa: E402
from xaig.core import registry  # noqa: E402
from xaig.core.errors import AdapterError, RequestError  # noqa: E402
from xaig.latents import (  # noqa: E402
    Hook,
    Intervenable,
    LatentSource,
    ReferenceFields,
    Rollout,
    open_source,
    read_latents,
    result_provenance,
)
from xaig.latents.grid import Grid  # noqa: E402
from xaig.latents.toy import write_toy  # noqa: E402

# -- the planted source ------------------------------------------------------------------
#
# Four times at six hours, in a year with a leap day and a calendar without one: from 28
# February to 1 March is six hours under `noleap` and a day and a quarter under a calendar
# that has a 29th. Every fifth node is masked and holds NaN, as land does. Values are
# multiples of a quarter below sixteen, so no storage dtype rounds them.

TIMES = (
    "0424-02-28T12:00:00",
    "0424-02-28T18:00:00",
    "0424-03-01T00:00:00",
    "0424-03-01T06:00:00",
)
FIELD_TIMES = ("0424-02-28T06:00:00", *TIMES)  # the state the first pass started from
N_LAT, N_LON, WIDTHS = 4, 6, (3, 5)
N_NODES = N_LAT * N_LON
STEP = 21600
EXPERIMENT = {"seed": 3, "note": "planted"}
NETWORK_LAYERS = (2, 6)


def _masked() -> np.ndarray:
    return np.arange(N_NODES) % 5 == 0


def planted_grid() -> Grid:
    lat, lon = np.meshgrid(np.linspace(-67.5, 67.5, N_LAT), np.arange(N_LON) * 60.0, indexing="ij")
    return Grid(lat=lat.ravel(), lon=lon.ravel(), shape=(N_LAT, N_LON), mask=~_masked())


def planted_layer(layer: int) -> np.ndarray:
    """``(n_times, n_nodes, width)``: a number for every place, NaN where masked."""
    t, n, c = np.meshgrid(
        np.arange(len(TIMES)), np.arange(N_NODES), np.arange(WIDTHS[layer]), indexing="ij"
    )
    values = ((t * 7 + n * 3 + c * 5 + layer) % 61) / 4.0
    values[:, _masked(), :] = np.nan
    return values


def planted_field() -> np.ndarray:
    s, n = np.meshgrid(np.arange(len(FIELD_TIMES)), np.arange(N_NODES), indexing="ij")
    values = s * 10.0 + n * 0.5
    values[:, _masked()] = np.nan
    return values


def _write_planted(adapter: str, path: Path, *, nan_at_valid: bool = False) -> Path:
    layers = [(f"level {i}", planted_layer(i)) for i in range(len(WIDTHS))]
    if nan_at_valid:
        layers[0][1][2, 1, 0] = np.nan  # a valid node, at the third time, that is not a number
    registry.get(adapter).write(
        path,
        grid=planted_grid(),
        times=TIMES,
        layers=layers,
        network_layers=NETWORK_LAYERS,
        fields={"t2m": planted_field()},
        field_times=FIELD_TIMES,
        model="planted-model",
        component="planted-part",
        checkpoint="planted.ckpt",
        calendar="noleap",
        timestep_seconds=STEP,
        experiment=EXPERIMENT,
    )
    return path


# -- the cases ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatentCase:
    """How to make a source of one adapter, and what to open it with. ``planted`` cases
    write the arrays above; ``masked`` says whether a found source has nodes marked invalid."""

    name: str
    adapter: str
    build: Callable[[Path], Path]
    options: dict[str, Any] = field(default_factory=dict)
    planted: bool = False
    masked: bool = False

    def open(self, path: Path, **more: Any):
        return open_source(path, adapter=self.adapter, **self.options, **more)


def _toy(adapter: str) -> Callable[[Path], Path]:
    return lambda tmp: write_toy(tmp / "toy", adapter=adapter)


LATENT_CASES = [
    LatentCase(
        "latent-archive/planted",
        "latent-archive",
        lambda tmp: _write_planted("latent-archive", tmp / "planted"),
        planted=True,
    ),  # fmt: skip
    LatentCase(
        "latent-archive/toy",
        "latent-archive",
        _toy("latent-archive"),
        options={"mask_variable": "sst"},
        masked=True,
    ),  # fmt: skip
    LatentCase(
        "latent-archive/handwritten",
        "latent-archive",
        lambda tmp: write_latent_archive(tmp / "hand"),
    ),
    LatentCase(
        "bundle-dir/planted",
        "bundle-dir",
        lambda tmp: _write_planted("bundle-dir", tmp / "planted"),
        planted=True,
    ),  # fmt: skip
    LatentCase("bundle-dir/toy", "bundle-dir", _toy("bundle-dir")),
    LatentCase(
        "bundle-dir/toy-unmasked", "bundle-dir", _toy("bundle-dir"), options={"unmasked": True}
    ),  # an option, so that options reaching the provenance is asked of it
]
PLANTED = [case for case in LATENT_CASES if case.planted]


@pytest.fixture(params=LATENT_CASES, ids=lambda case: case.name)
def case(request) -> LatentCase:
    return request.param


@pytest.fixture(params=PLANTED, ids=lambda case: case.name)
def planted(request) -> LatentCase:
    return request.param


@pytest.fixture
def source(case, tmp_path):
    return case.open(case.build(tmp_path))


@pytest.fixture
def known(planted, tmp_path):
    return planted.open(planted.build(tmp_path))


def _same(a, b) -> bool:
    return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))


# -- any source: the protocol, shapes, selection ------------------------------------------


def test_a_source_is_a_latent_source_that_says_what_it_holds(source):
    assert isinstance(source, LatentSource)
    info, grid = source.info(), source.grid()
    assert info.n_nodes == grid.n_nodes > 0
    assert info.times and all(isinstance(t, str) for t in info.times)
    assert len(set(info.times)) == len(info.times)
    assert info.layers and len({layer.index for layer in info.layers}) == len(info.layers)
    assert info.last_layer == max(layer.index for layer in info.layers)
    for layer in info.layers:
        assert info.layer(layer.index) == layer and layer.n_channels > 0
    with pytest.raises(RequestError, match="no layer"):
        info.layer(max(layer.index for layer in info.layers) + 1)
    assert set(info.identity()) == {"model", "component", "checkpoint"}  # all three, declared


def test_a_load_is_float32_nodes_by_channels_in_the_order_asked(source):
    info, layer = source.info(), source.info().layers[-1]
    full = source.load(0, layer.index)
    assert full.dtype == np.float32 and full.shape == (info.n_nodes, layer.n_channels)
    nodes, channels = [info.n_nodes - 1, 1, 3], [layer.n_channels - 1, 0]
    assert _same(source.load(0, layer.index, nodes=nodes), full[nodes])
    assert _same(source.load(0, layer.index, channels=channels), full[:, channels])
    both = source.load(0, layer.index, channels=channels, nodes=nodes)
    assert both.shape == (3, 2) and _same(both, full[np.ix_(nodes, channels)])
    assert _same(source.load(0, layer.index, nodes=np.array(nodes)), full[nodes])  # numpy too


def test_a_time_is_a_label_or_a_position(source):
    info, layer = source.info(), source.info().layers[0].index
    for position in (0, len(info.times) - 1, -1):
        by_label = source.load(info.times[position], layer)
        assert _same(by_label, source.load(position, layer))


def test_what_is_returned_is_the_callers_to_modify(source):
    layer = source.info().layers[0].index
    first = source.load(0, layer)
    kept = first.copy()
    first[:] = -999.0
    assert _same(source.load(0, layer), kept)


def test_a_request_it_cannot_meet_is_a_request_error(source):
    info = source.info()
    layer = info.layers[0]
    with pytest.raises(RequestError):
        source.load(0, layer.index, nodes=[info.n_nodes])
    with pytest.raises(RequestError):
        source.load(0, layer.index, channels=[layer.n_channels])
    with pytest.raises(RequestError, match="no layer"):
        source.load(0, info.last_layer + 1)
    with pytest.raises(RequestError):
        source.load("not a time", layer.index)
    with pytest.raises(RequestError):
        source.load(len(info.times), layer.index)


def test_a_negative_index_is_refused_and_not_counted_from_the_end(source):
    layer = source.info().layers[0]
    with pytest.raises(RequestError, match="channel index"):
        source.load(0, layer.index, channels=[0, -1])
    with pytest.raises(RequestError, match="node index"):
        source.load(0, layer.index, nodes=[-1])


def test_the_grid_marks_what_means_nothing_and_weighs_the_rest(case, source):
    grid = source.grid()
    assert grid.valid.dtype == bool and grid.valid.shape == (grid.n_nodes,)
    assert grid.lat.shape == grid.lon.shape == (grid.n_nodes,)
    assert grid.valid.any()
    assert bool((~grid.valid).any()) == (case.masked or case.planted)
    weights = grid.weights()
    assert weights.sum() == pytest.approx(1.0) and not weights[~grid.valid].any()
    assert (weights[grid.valid] > 0).all()


def test_times_carry_the_calendar_they_are_labels_of(source):
    info = source.info()
    elapsed = info.elapsed_seconds()
    assert info.calendar and info.timestep_seconds
    assert elapsed is not None and elapsed[0] == 0.0
    assert all(b > a for a, b in zip(elapsed, elapsed[1:], strict=False))


def test_provenance_names_the_network_the_options_and_this_xaig(case, source):
    info = source.info()
    provenance = info.provenance()
    assert {"source", "model", "component", "checkpoint"} <= provenance.keys()
    assert provenance["model"] and provenance["checkpoint"]
    assert provenance.get("options", {}) == case.options
    assert "revision" not in provenance  # a local source has no commit to name
    assert result_provenance(info)["xaig"]


def test_the_same_source_opened_twice_says_and_gives_the_same(case, tmp_path):
    path = case.build(tmp_path)
    a, b = case.open(path), case.open(path)
    assert a.info() == b.info()
    assert _same(a.grid().lat, b.grid().lat) and _same(a.grid().valid, b.grid().valid)
    assert _same(a.grid().weights(), b.grid().weights())
    for layer in a.info().layers:
        assert _same(a.load(-1, layer.index), b.load(-1, layer.index))
        assert _same(a.load(-1, layer.index), a.load(-1, layer.index))


def test_reference_fields_are_numbers_per_node_where_offered(source):
    if not isinstance(source, ReferenceFields):
        pytest.skip("this adapter keeps no physical fields")
    names = source.field_names()
    assert isinstance(names, tuple)
    for name in names:
        values = source.field(name, 0)
        assert values.dtype == np.float64 and values.shape == (source.info().n_nodes,)
    with pytest.raises(RequestError, match="no field"):
        source.field("not-a-field", 0)


# -- the options, and what a broken source is ----------------------------------------------


def test_an_option_the_adapter_does_not_declare_is_refused_by_name(case, tmp_path):
    path = case.build(tmp_path)
    with pytest.raises(AdapterError, match=r"does not accept option\(s\) no_such_option; accepted"):
        case.open(path, no_such_option=1)
    with pytest.raises(AdapterError, match="needs a source"):
        registry.create(case.adapter, options=case.options)


def test_a_source_that_is_not_there_or_not_one_is_an_adapter_error(case, tmp_path):
    with pytest.raises(AdapterError):
        case.open(tmp_path / "absent")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(AdapterError):
        case.open(empty)
    stray = tmp_path / "file.txt"
    stray.write_text("not a source")
    with pytest.raises(AdapterError):
        case.open(stray)


# -- planted: what was written is what comes back ------------------------------------------


def test_what_was_written_is_what_is_read(known):
    info = known.info()
    assert info.times == TIMES
    assert (info.model, info.component, info.checkpoint) == (
        "planted-model", "planted-part", "planted.ckpt",
    )  # fmt: skip
    assert info.calendar == "noleap" and info.timestep_seconds == STEP
    assert [layer.n_channels for layer in info.layers] == list(WIDTHS)
    assert [layer.position for layer in info.layers] == list(NETWORK_LAYERS)
    assert all(layer.index == i for i, layer in enumerate(info.layers))
    for index in range(len(WIDTHS)):
        expected = planted_layer(index)
        for t in range(len(TIMES)):
            assert _same(known.load(t, index), expected[t])
    grid, truth = known.grid(), planted_grid()
    assert (
        grid.shape == (N_LAT, N_LON) and _same(grid.lat, truth.lat) and _same(grid.lon, truth.lon)
    )
    assert _same(grid.valid, ~_masked())


def test_a_calendar_without_a_leap_day_puts_one_six_hour_step_across_february(known):
    """A reader that gave 1424 its 29th of February would put a day and a quarter here."""
    assert known.info().elapsed_seconds() == (0.0, STEP, 2 * STEP, 3 * STEP)


def test_the_exporters_notes_and_the_options_reach_the_provenance(known):
    assert known.info().provenance()["experiment"] == EXPERIMENT


def test_a_masked_node_may_hold_nan_and_a_valid_one_may_not(planted, tmp_path):
    clean = planted.open(planted.build(tmp_path / "clean"))
    assert np.isnan(clean.load(0, 0)[_masked()]).all()  # land holds what the model wrote
    values = read_latents(clean, 0, 0, nodes=[1, 2, 3])
    assert np.isfinite(values).all()
    assert np.isfinite(read_latents(clean, 3, 1)[~_masked()]).all()  # and read_latents accepts it

    dirty = _write_planted(planted.adapter, tmp_path / "dirty", nan_at_valid=True)
    broken = planted.open(dirty)
    with pytest.raises(RequestError, match="not finite") as refused:
        read_latents(broken, 2, 0)
    assert "layer 0" in str(refused.value) and TIMES[2] in str(refused.value)
    assert np.isfinite(read_latents(broken, 1, 0)[~_masked()]).all()  # another time is fine


def test_physical_fields_come_back_at_their_times_and_leads(known):
    assert isinstance(known, ReferenceFields) and known.field_names() == ("t2m",)
    expected = planted_field()
    assert _same(known.field("t2m", TIMES[0]), expected[1])
    assert _same(known.field("t2m", 0, lead=1), expected[2])
    assert _same(known.field("t2m", -1), expected[-1])
    with pytest.raises(RequestError, match="ends? before"):
        known.field("t2m", -1, lead=1)


# -- the registry: shipped adapters are all under contract ---------------------------------


def _shipped() -> dict[str, type]:
    return {
        ep.name: ep.load()
        for ep in entry_points(group="xaig.adapters")
        if ep.dist is not None and ep.dist.name == "xaig"
    }


def test_every_shipped_adapter_is_under_contract():
    """A new adapter module, registered, must be added to a case list here, so that adding
    one means passing the contract and not remembering to."""
    latent = {c.adapter for c in LATENT_CASES}
    runnable = {name for name, _ in INTERVENABLE_CASES}
    for name, factory in _shipped().items():
        reads = all(hasattr(factory, m) for m in ("info", "grid", "load"))
        runs = all(hasattr(factory, m) for m in ("info", "grid", "initial_state", "run"))
        assert reads or runs, f"{name} implements no protocol xaig knows"
        assert not reads or name in latent, f"{name} is a LatentSource with no case in LATENT_CASES"
        assert not runs or name in runnable, f"{name} is Intervenable with no INTERVENABLE_CASES"
    assert {"latent-archive", "bundle-dir", "toy-dynamics"} <= set(_shipped())


def test_a_planted_case_needs_a_writer():
    for case in PLANTED:
        assert callable(getattr(registry.get(case.adapter), "write", None)), case.name


# -- Intervenable ----------------------------------------------------------------------------

INTERVENABLE_CASES = [("toy-dynamics", {"masked": 2})]


@pytest.fixture(params=INTERVENABLE_CASES, ids=lambda c: c[0])
def system(request):
    name, options = request.param
    built = registry.create(name, options=options)
    assert isinstance(built, Intervenable)
    return built


def _place(system) -> tuple[int, int]:
    return system.info().layers[-1].index, 1


def test_a_system_runs_and_writes_fields_that_are_nan_only_where_masked(system):
    n, grid = system.info().n_nodes, system.grid()
    rollout = system.run(system.initial_state(), 3, noise_seed=1)
    assert isinstance(rollout, Rollout) and rollout.fields
    for values in rollout.fields.values():
        assert values.shape == (3, n)
        assert np.isnan(values[:, ~grid.valid]).all() and np.isfinite(values[:, grid.valid]).all()
    assert not rollout.latents  # nothing was asked to be recorded


def test_recorded_latents_are_float32_at_exactly_the_places_asked(system):
    info = system.info()
    places = [(info.layers[0].index, 0), _place(system)]
    rollout = system.run(system.initial_state(), 3, noise_seed=1, record=places)
    assert set(rollout.latents) == set(places)
    for (layer, _), values in rollout.latents.items():
        assert values.dtype == np.float32 and values.shape == (
            info.n_nodes,
            info.layer(layer).n_channels,
        )


def test_noise_is_a_function_of_the_seed_alone(system):
    state = system.initial_state()
    a = system.run(state, 3, noise_seed=4)
    assert all(_same(a.fields[k], system.run(state, 3, noise_seed=4).fields[k]) for k in a.fields)
    other = system.run(state, 3, noise_seed=5)
    assert any(not _same(a.fields[k], other.fields[k]) for k in a.fields)
    quiet = system.run(state, 3)  # no seed: no noise, and the same every time
    assert all(_same(quiet.fields[k], system.run(state, 3).fields[k]) for k in quiet.fields)


def test_running_does_not_change_the_state_it_was_given(system):
    state = system.initial_state()
    kept = state.copy()
    system.run(state, 3, noise_seed=1)
    assert _same(state, kept)


def test_a_hook_sees_what_the_pass_produced_and_what_it_returns_is_continued_with(system):
    layer, time = _place(system)
    state, seen = system.initial_state(), []
    control = system.run(state, 3, noise_seed=2, record=[(layer, time)])

    def watch(latents):
        seen.append(latents.copy())
        return latents

    same = system.run(state, 3, [Hook(layer, time, watch)], noise_seed=2)
    assert len(seen) == 1 and _same(seen[0], control.latents[(layer, time)])
    assert seen[0].dtype == np.float32
    assert all(_same(same.fields[k], control.fields[k]) for k in control.fields)  # a no-op is one

    pushed = system.run(state, 3, [Hook(layer, time, lambda h: h + 1.0)], noise_seed=2)
    moved = [k for k in control.fields if not _same(pushed.fields[k], control.fields[k])]
    assert moved, "an edit that changed the latents changed nothing downstream"
    for k in moved:  # and nothing before it: the edit and the noise are independent
        before = np.nan_to_num(pushed.fields[k][:time] - control.fields[k][:time])
        assert not before.any()


def test_a_place_that_does_not_exist_is_refused_before_anything_is_returned(system):
    layer, time = _place(system)
    nowhere = max(entry.index for entry in system.info().layers) + 1
    state = system.initial_state()
    with pytest.raises(RequestError):
        system.run(state, 3, [Hook(nowhere, 0, lambda h: h)])
    with pytest.raises(RequestError):
        system.run(state, 3, [Hook(layer, 3, lambda h: h)])  # a run of 3 has times 0 to 2
    with pytest.raises(RequestError):
        system.run(state, 3, record=[(nowhere, 0)])
    with pytest.raises(RequestError):
        system.run(state, 3, record=[(layer, 3)])
    with pytest.raises(RequestError):
        system.run(state, 0)


def test_a_system_reads_no_source_so_it_takes_options_by_keyword_and_refuses_the_rest(system):
    name = next(n for n, _ in INTERVENABLE_CASES)
    with pytest.raises(AdapterError, match=r"does not accept option\(s\) no_such_option"):
        registry.create(name, options={"no_such_option": 1})
