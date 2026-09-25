"""``xaig taig``: train a sparse autoencoder. Must import on a base install, so
torch and numpy are imported inside the command that needs them."""

from __future__ import annotations

import click

from xaig.core.extras import require


@click.group(name="taig")
def taig() -> None:
    """Train a sparse autoencoder on a model's latents."""


@taig.command("sae")
@click.argument("source", type=click.Path())
@click.option("--adapter", default="latent-archive", show_default=True, help="How SOURCE is read.")
@click.option(
    "--mask-variable",
    help="Reference-file variable that is missing where nodes mean nothing (e.g. sst).",
)
@click.option("--layer", type=int, help="Layer to read.  [default: the last]")
@click.option("--target-layer", type=int, help="Write this layer instead: a transcoder.")
@click.option("--features", "n_features", type=int, default=1024, show_default=True)
@click.option(
    "--activation",
    type=click.Choice(["relu", "topk", "bspline"]),
    default="topk",
    show_default=True,
)
@click.option("--k", type=int, default=32, show_default=True, help="Active features, for topk.")
@click.option("--l1", type=float, default=5.0, show_default=True, help="For relu and bspline.")
@click.option(
    "--node-norm",
    is_flag=True,
    help="Centre each node over its channels and scale it to unit RMS first.",
)
@click.option("--epochs", type=int, default=2, show_default=True)
@click.option("--batch-size", type=int, default=4096, show_default=True)
@click.option("--lr", type=float, default=1e-3, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--time", "times", multiple=True, help="Time label or position; repeatable.")
@click.option("--device", help="cpu, cuda or mps.  [default: the best there is]")
@click.option("--out", type=click.Path(dir_okay=False), required=True, help="Basis file to write.")
def sae_cmd(source, adapter, mask_variable, layer, out, times, k, activation, **kw) -> None:
    """Fit a sparse autoencoder to one layer of SOURCE; write a basis file.

    The file is used wherever a PCA is: `xaig daig latent region --basis`,
    `latent series --basis`, `latent fields --basis`.
    """
    # Asked for first: whoever wants to train should be pointed at the one extra that
    # brings everything, not at numpy's and then ours.
    require("torch", "taig")
    from xaig.daig.latent import open_source, parse_time, save_basis
    from xaig.taig.train import fit_sae

    options = {"mask_variable": mask_variable} if mask_variable else {}
    opened = open_source(source, adapter=adapter, **options)
    layer = opened.info().last_layer if layer is None else layer

    def progress(step: int, error: float) -> None:
        click.echo(f"  step {step:6d}  reconstruction {error:.4g}", err=True)

    dictionary = fit_sae(
        opened,
        layer=layer,
        activation=activation,
        k=k if activation == "topk" else None,
        times=[parse_time(t) for t in times] or None,
        progress=progress,
        **kw,
    )
    save_basis(out, dictionary)
    metrics = dictionary.meta["metrics"]
    click.echo(
        f"wrote {out}: {dictionary.n_features} feature(s) of layer {layer}; "
        f"{100 * metrics['explained_variance']:.1f}% of the variance explained, "
        f"{metrics['mean_active_features']:.1f} active per node, "
        f"{100 * metrics['dead_fraction']:.1f}% dead"
    )


__all__ = ["taig"]
