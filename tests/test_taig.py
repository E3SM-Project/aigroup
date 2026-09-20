"""taig: the blocks, and fitting one to a model's latents."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from xaig._cli import cli


def test_the_command_is_listed_and_explains_itself_without_torch():
    """`xaig --help` imports every cli module, torch or no torch."""
    listed = CliRunner().invoke(cli, ["taig", "sae", "--help"])
    assert listed.exit_code == 0 and "sparse autoencoder" in listed.output


def test_without_torch_the_command_names_the_one_extra_that_brings_everything():
    """Not numpy's extra first and ours second."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "sys.modules['torch'] = sys.modules['numpy'] = None\n"  # as if neither were installed
        "from click.testing import CliRunner\n"
        "from xaig._cli import cli\n"
        "print(CliRunner().invoke(cli, ['taig', 'sae', 'anywhere', '--out', 'x.npz']).output)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout
    assert "'taig' extra" in out and "'daig' extra" not in out and "Traceback" not in out


np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from conftest import BUMP, N_CHANNELS  # noqa: E402
from xaig.core.errors import RequestError  # noqa: E402
from xaig.daig.latent import (  # noqa: E402
    Decomposition,
    Region,
    analyse_region,
    bspline_activation,
    load_basis,
    open_source,
)
from xaig.taig.sae import BSplineActivation, SparseAutoencoder  # noqa: E402
from xaig.taig.train import fit_sae  # noqa: E402

HERE = Region(lat=BUMP[0], lon=BUMP[1], radius_km=2500.0)


def test_the_spline_starts_as_a_relu_and_is_learnable():
    spline = BSplineActivation(n_features=5, n_intervals=8, upper=6.0)
    z = torch.linspace(-3.0, 9.0, 200)[:, None].repeat(1, 5).requires_grad_()
    out = spline(z)
    assert torch.allclose(out, torch.relu(z), atol=1e-5)
    out.sum().backward()
    assert spline.coefficients.grad.abs().sum() > 0 and z.grad.abs().sum() > 0


def test_torch_and_numpy_evaluate_a_bent_spline_alike():
    """One trains it, the other uses it; they must be the same function."""
    torch.manual_seed(0)
    spline = BSplineActivation(n_features=4, n_intervals=6, upper=5.0)
    with torch.no_grad():
        spline.coefficients.add_(torch.randn_like(spline.coefficients))
    z = torch.linspace(-2.0, 8.0, 301)[:, None].repeat(1, 4)
    ours = bspline_activation(z.numpy(), spline.coefficients.detach().numpy(), 5.0)
    assert ours == pytest.approx(spline(z).detach().numpy(), abs=1e-5)


@pytest.mark.parametrize("activation", ["relu", "topk", "bspline"])
def test_a_block_and_its_dictionary_encode_alike(activation):
    torch.manual_seed(1)
    block = SparseAutoencoder(6, 20, activation=activation, k=4 if activation == "topk" else None)
    if activation == "bspline":
        with torch.no_grad():
            block.spline.coefficients.add_(0.3 * torch.randn_like(block.spline.coefficients))
    mean, scale = np.arange(6, dtype=np.float32), 2.5
    dictionary = block.to_dictionary(input_mean=mean, input_scale=scale, note="x")
    assert isinstance(dictionary, Decomposition) and dictionary.meta == {"note": "x"}

    latents = np.random.default_rng(0).normal(size=(50, 6)).astype(np.float32) * 3
    standardised = torch.as_tensor((latents - mean) / scale)
    rebuilt, features = block(standardised)
    assert dictionary.transform(latents) == pytest.approx(features.detach().numpy(), abs=1e-5)
    back = rebuilt.detach().numpy() * scale + mean
    assert dictionary.reconstruct(latents) == pytest.approx(back, abs=1e-4)
    if activation == "topk":
        assert (dictionary.transform(latents) > 0).sum(axis=1).max() <= 4


def test_a_block_that_cannot_be_is_refused():
    with pytest.raises(ValueError, match="needs 1 <= k"):
        SparseAutoencoder(6, 20, activation="topk")
    with pytest.raises(ValueError, match="one of relu, topk, bspline"):
        SparseAutoencoder(6, 20, activation="gelu")


def test_the_penalty_is_what_makes_a_relu_code_sparse():
    torch.manual_seed(0)
    block, x = SparseAutoencoder(6, 20), torch.randn(64, 6)
    plain, error, _ = block.loss(x)
    penalised, same_error, features = block.loss(x, l1=2.0)
    assert plain == error == same_error and penalised > error
    top = SparseAutoencoder(6, 20, activation="topk", k=3)
    total, error, features = top.loss(x, l1=2.0)
    assert total == error and (features > 0).sum(-1).max() <= 3  # topk needs no penalty


def test_fitting_finds_the_planted_bump(latent_archive):
    """Six channels, one of which holds all the structure: a few features must
    rebuild the layer, and the one that answers in the region must be made of
    channel 4."""
    source = open_source(latent_archive)
    seen = []
    dictionary = fit_sae(
        source, layer=2, n_features=12, activation="topk", k=3, epochs=60, batch_size=96,
        lr=3e-3, device="cpu", progress=lambda step, error: seen.append(step),
    )  # fmt: skip
    metrics = dictionary.meta["metrics"]
    assert metrics["explained_variance"] > 0.8 and metrics["mean_active_features"] <= 3
    assert dictionary.meta["fitted_on"]["layer"] == 2 and seen and seen[0] == 25
    assert dictionary.input_mean[1] == pytest.approx(50.0, abs=0.01)  # standardised, not raw

    result = analyse_region(source, time=0, layer=2, region=HERE, n_components=1, basis=dictionary)
    assert result.feature_info[0]["top_loadings"][0]["channel"] == 4
    centre = source.grid().nearest(*BUMP)
    assert result.scores[centre, 0] == pytest.approx(np.nanmax(result.scores[:, 0]))


def test_a_transcoder_writes_another_layer(latent_archive):
    dictionary = fit_sae(
        open_source(latent_archive), layer=0, target_layer=2, n_features=8, activation="relu",
        l1=0.01, epochs=2, batch_size=96, device="cpu",
    )  # fmt: skip
    assert dictionary.meta["fitted_on"]["target_layer"] == 2
    assert dictionary.output_mean is not None and dictionary.output_scale != dictionary.input_scale
    assert dictionary.reconstruct(np.zeros((3, N_CHANNELS))).shape == (3, N_CHANNELS)


def test_a_fit_that_asks_for_nothing_is_refused(latent_archive):
    with pytest.raises(RequestError, match="n_features"):
        fit_sae(open_source(latent_archive), layer=2, n_features=0)


def test_cli_fits_and_the_file_is_a_basis_anywhere(latent_archive, tmp_path):
    out = tmp_path / "sae.npz"
    args = ["taig", "sae", str(latent_archive), "--features", "8", "--k", "2", "--epochs", "3"]
    fitted = CliRunner().invoke(cli, [*args, "--batch-size", "96", "--device", "cpu", "--out", out])
    assert fitted.exit_code == 0, fitted.output
    assert "8 feature(s) of layer 2" in fitted.output and "active per node" in fitted.output
    assert load_basis(out).meta["training"]["activation"] == "topk"
    region = ["daig", "latent", "region", str(latent_archive), "--lat", "7.5", "--lon", "45"]
    shown = CliRunner().invoke(
        cli, [*region, "--radius-km", "2500", "--features", "2", "--basis", out]
    )
    assert shown.exit_code == 0 and "peak" in shown.output, shown.output


def test_torch_and_numpy_break_a_tie_the_same_way():
    """Two features with one encoder row always tie; whichever wins, both
    implementations must name the same one, and neither may keep more than k."""
    torch.manual_seed(2)
    block = SparseAutoencoder(6, 10, activation="topk", k=3)
    with torch.no_grad():
        block.encoder.weight[1:4] = block.encoder.weight[0]
        block.encoder.bias[:4] = 0.0
    dictionary = block.to_dictionary(input_mean=np.zeros(6, dtype=np.float32))
    latents = np.random.default_rng(0).normal(size=(200, 6)).astype(np.float32)
    features = block.encode(torch.as_tensor(latents)).detach().numpy()
    assert (features > 0).sum(axis=1).max() == 3  # never the four that tie
    assert np.array_equal(dictionary.transform(latents) > 0, features > 0)


def test_a_fitted_features_direction_has_unit_length(latent_archive):
    """Otherwise an activation is in units of its own, and two features cannot
    be compared by it."""
    dictionary = fit_sae(
        open_source(latent_archive), layer=2, n_features=8, activation="relu", l1=0.1,
        epochs=5, batch_size=96, device="cpu",
    )  # fmt: skip
    assert np.linalg.norm(dictionary.decoder, axis=1) == pytest.approx(1.0, abs=1e-5)
