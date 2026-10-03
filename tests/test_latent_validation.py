"""One behaviour for what cannot be had: activations that are not numbers where the
grid says they must be, and selections that are empty. Each is a ``RequestError``
that says what and where, and never a NaN that surfaces three steps later."""

from __future__ import annotations

import re

import pytest

np = pytest.importorskip("numpy")

from click.testing import CliRunner  # noqa: E402

from conftest import MemorySource  # noqa: E402
from xaig._cli import cli  # noqa: E402
from xaig.core.errors import RequestError  # noqa: E402
from xaig.latents import (  # noqa: E402
    Box,
    Region,
    accumulate_moments,
    analyse_region,
    difference,
    difference_growth,
    iter_batches,
    load_channels,
    read_latents,
    region_series,
)

WHOLE = Box(-90.0, 90.0, 0.0, 360.0)
N_NODES = 48


def _planted(poison=None, mask=None):
    """Two layers, three times, four channels; ``poison`` is ``{(layer, time): value}``
    written at node 5 of channel 2."""
    rng = np.random.default_rng(0)
    data = {layer: rng.normal(0.0, 1.0, (3, N_NODES, 4)).astype(np.float32) for layer in (0, 1)}
    for (layer, time), value in (poison or {}).items():
        data[layer][time, 5, 2] = value
    return MemorySource(data, mask=mask)


NAMED = r"layer 1 at t2 holds 1 valid node\(s\).*channel 2.*in memory"


def test_read_latents_names_the_layer_the_time_and_the_channel():
    source = _planted({(1, 2): np.nan})
    with pytest.raises(RequestError, match=NAMED):
        read_latents(source, 2, 1)
    with pytest.raises(RequestError, match=NAMED):
        read_latents(source, "t2", 1, channels=[0, 2])
    assert read_latents(source, 2, 1, channels=[0, 1]).shape == (N_NODES, 2)  # not in these
    assert read_latents(source, 2, 1, nodes=[0, 1, 4]).shape == (3, 4)  # nor at these nodes
    assert read_latents(source, 1, 1).shape == (N_NODES, 4)  # nor at other times


def test_a_node_the_mask_removes_may_hold_anything():
    mask = np.arange(N_NODES) != 5
    source = _planted({(1, 2): np.nan}, mask=mask)
    assert np.isnan(read_latents(source, 2, 1)[5, 2])  # handed over as it is...
    moments = accumulate_moments(source, layer=1)  # ...and left out of what counts
    assert np.isfinite(moments.covariance).all() and moments.times == ("t0", "t1", "t2")


def test_every_reader_of_latents_refuses_a_valid_node_that_is_not_a_number():
    source = _planted({(1, 2): np.nan})
    other = _planted()
    calls = {
        "moments": lambda: accumulate_moments(source, layer=1),
        "batches": lambda: list(  # nodes are drawn, so give the poisoned one many chances
            iter_batches(source, layer=1, batch_size=N_NODES, times=[2], epochs=40)
        ),
        "region": lambda: analyse_region(source, time=2, layer=1, region=WHOLE),
        "series": lambda: region_series(source, layer=1, region=WHOLE, channels=[2]),
        "channels": lambda: load_channels(source, time=2, layer=1, channels=[2]),
        "difference": lambda: difference(other, source, time=2, layer=1),
        "growth": lambda: difference_growth(other, source),
        "noise": lambda: difference_growth(other, other, noise=source),
    }
    accepted = []
    for name, call in calls.items():
        try:
            call()
        except RequestError as error:
            assert re.search(NAMED, str(error)), (name, str(error))
        else:
            accepted.append(name)
    assert not accepted


def test_nothing_selected_is_refused_saying_what_was_empty():
    source, nowhere = _planted(), Region(0.0, 0.0, 1.0)
    empty = {
        "no times selected": lambda: accumulate_moments(source, layer=0, times=[]),
        "no valid nodes within": lambda: analyse_region(source, time=0, layer=0, region=nowhere),
        "no valid nodes within 1 km": lambda: region_series(
            source, layer=0, region=nowhere, channels=[0]
        ),
        "no valid nodes: the mask": lambda: accumulate_moments(
            _planted(mask=np.zeros(N_NODES, bool)), layer=0
        ),
    }
    for message, call in empty.items():
        with pytest.raises(RequestError, match=message):
            call()


def test_the_cli_says_it_in_one_line(latent_archive):
    """A NaN at a valid node of an archive on disk, through the command line: an
    error line naming the layer and time, no traceback."""
    step = latent_archive / "step_02.npy"
    data = np.load(step)
    data[0, 3, 4] = np.nan
    np.save(step, data)
    out = latent_archive / "pca.npz"
    shown = CliRunner().invoke(
        cli, ["latents", "pca", str(latent_archive), "--layer", "2", "--out", str(out)]
    )
    assert shown.exit_code == 1
    assert "layer 2 at 0425-01-01T06:00:00 holds 1 valid node(s)" in shown.output
    assert "Traceback" not in shown.output
