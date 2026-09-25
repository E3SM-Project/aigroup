"""The parts of the ACE exporter that know nothing of ACE: which times are kept,
which activations are recorded, and in what node order. The rest needs ACE's
environment and a checkpoint, and is checked by running it."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from xaig.adapters.ace_export import (  # noqa: E402
    BlockRecorder,
    find_blocks,
    keep_every,
    layer_names,
    to_nodes,
)
from xaig.core.errors import RequestError  # noqa: E402


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.mix = nn.Conv2d(width, width, 1)

    def forward(self, x, context):
        return x + self.mix(x) * context


class _Net(nn.Module):
    """A residual stack like the SFNO's: an encoder, blocks taking a context,
    a decoder."""

    def __init__(self, width: int = 3, n_blocks: int = 2) -> None:
        super().__init__()
        self.embed_dim = width
        self.encoder = nn.Conv2d(2, width, 1)
        self.blocks = nn.ModuleList(_Block(width) for _ in range(n_blocks))
        self.decoder = nn.Conv2d(width, 2, 1)

    def forward(self, x, context=1.0):
        x = self.encoder(x)
        for block in self.blocks:
            x = block(x, context)
        return self.decoder(x)


class _Wrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module = _Net()

    def forward(self, x):
        return self.module(x)


def test_the_stream_is_every_block_input_and_the_last_output():
    torch.manual_seed(0)
    wrapped = _Wrapper()
    network, blocks = find_blocks(wrapped)
    assert network is wrapped.module and network.embed_dim == 3
    x = torch.randn(2, 2, 4, 5)
    with torch.no_grad(), BlockRecorder(blocks) as recorder:
        wrapped(x)
        taken = recorder.take()
        # by hand, the same stream
        h = network.encoder(x)
        expected = [h]
        for block in network.blocks:
            h = block(h, 1.0)
            expected.append(h)
    assert len(taken) == 3
    for got, want in zip(taken, expected, strict=True):
        assert torch.equal(got, want)
    assert not blocks[0]._forward_pre_hooks  # nothing outlives the recording


def test_a_second_call_before_taking_is_refused():
    network = _Net()
    with torch.no_grad(), BlockRecorder(network.blocks):
        network(torch.randn(1, 2, 3, 3))
        with pytest.raises(RuntimeError, match="more than once per step"):
            network(torch.randn(1, 2, 3, 3))


def test_nodes_are_in_the_grids_row_major_order():
    # (channels, lat, lon) with each value naming its own place
    lat, lon, channels = 3, 4, 2
    tensor = torch.arange(channels * lat * lon, dtype=torch.float32).reshape(channels, lat, lon)
    nodes = to_nodes(tensor, (lat, lon))
    assert nodes.shape == (lat * lon, channels) and nodes.dtype == np.float32
    # node i*lon + j is (lat i, lon j), as a meshgrid(indexing="ij") grid is flattened
    assert np.array_equal(nodes[1 * lon + 2], tensor[:, 1, 2].numpy())
    padded = torch.nn.functional.pad(tensor, (0, 2, 0, 1))  # a network's padding is cut
    assert np.array_equal(to_nodes(padded, (lat, lon)), nodes)


def test_kept_times_step_round_the_diurnal_cycle():
    kept = keep_every(n_times=1000, first=3, stop=200, every=29)
    assert kept[:3] == [3, 32, 61] and kept[-1] < 200
    assert {k % 4 for k in kept} == {0, 1, 2, 3}  # four six-hourly steps a day, all hit
    with pytest.raises(RequestError):
        keep_every(10, 0, 10, 0)


def test_layers_are_named_for_where_they_are():
    names = layer_names(8)
    assert len(names) == 9
    assert names[0] == ("block 0 input (residual stream)", "block_00_input")
    assert names[-1] == ("block 7 output (decoder input)", "block_07_output")
