# daig — diagnostics

Emulator-vs-reference evaluation: bias and time-mean maps, spectra, zonal means, drift.

**Skeleton in 0.1.0.** See the module docstring for the intended API.

## Rules that must not be lost when this is built

- **Area-weight everything.** Unweighted means on a Gaussian grid are simply wrong.
- **Be NaN-aware.** Ocean channels are undefined over land (~31% of points in the
  E3SMv3 configuration); a plain `mean` silently returns NaN or a biased number.
- Reading fields is an adapter's job (`MetricSource`/`ArtifactStore`). This subpackage
  computes and plots; it does not know file formats.
- Plotting requires the `viz` extra. Do not import matplotlib at package import time.
- May not import `caig` or `taig`.
