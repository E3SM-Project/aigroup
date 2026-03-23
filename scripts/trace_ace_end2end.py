#!/usr/bin/env python
"""Standalone script to torch.jit.trace a full ACE model for single-step inference.

Bundles the neural network, normalization, denormalization, residual prediction,
secondary decoder, and all atmosphere correctors into a single self-contained
TorchScript model.  The only import from the ``fme`` package is the checkpoint
loader (``load_stepper``); everything else is defined here so the traced .pt
file has zero runtime dependency on ``fme``.

Usage::

    python scripts/trace_ace_model.py checkpoint.ckpt [output_base] [--device cpu]

Outputs::

    {output_base}.pt              — TorchScript model
    {output_base}_metadata.yaml   — channel names, shapes, corrector config

Traced model signature::

    output = model(inputs)
    inputs:  [B, C_in + C_forcing, H, W]  concatenated (state, forcing)
    output:  [B, C_out, H, W]             denormalized, corrected output

The first C_in channels are the denormalized prognostic/diagnostic state;
the remaining C_forcing channels are the denormalized next-step forcing.
If no correctors are active the forcing channels are unused (pass zeros).

From C++::

    auto model = torch::jit::load("ace_traced.pt");
    auto out = model.forward(inputs).toTensor();
"""

from __future__ import annotations

import argparse
import logging
import pathlib
from typing import Any

import yaml

import torch
from torch import Tensor, nn

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (inlined from fme.core.constants / atmosphere_data)
# ---------------------------------------------------------------------------

GRAVITY = 9.80665  # m/s^2
LATENT_HEAT_OF_VAPORIZATION = 2.5e6  # J/kg

# Maps standard atmosphere field names to the prefixes used in ACE channel
# naming.  Copied from fme.core.atmosphere_data so this script is standalone.
ATMOSPHERE_FIELD_NAME_PREFIXES: dict[str, list[str]] = {
    "specific_total_water": ["specific_total_water_"],
    "surface_pressure": ["PRESsfc", "PS"],
    "surface_height": ["HGTsfc"],
    "surface_geopotential": ["PHIS"],
    "tendency_of_total_water_path_due_to_advection": [
        "tendency_of_total_water_path_due_to_advection"
    ],
    "latent_heat_flux": ["LHTFLsfc", "LHFLX"],
    "sensible_heat_flux": ["SHTFLsfc", "SHFLX"],
    "precipitation_rate": ["PRATEsfc", "surface_precipitation_rate"],
    "sfc_down_sw_radiative_flux": ["DSWRFsfc", "FSDS"],
    "sfc_up_sw_radiative_flux": ["USWRFsfc", "surface_upward_shortwave_flux"],
    "sfc_down_lw_radiative_flux": ["DLWRFsfc", "FLDS"],
    "sfc_up_lw_radiative_flux": ["ULWRFsfc", "surface_upward_longwave_flux"],
    "toa_up_lw_radiative_flux": ["ULWRFtoa", "FLUT"],
    "toa_up_sw_radiative_flux": ["USWRFtoa", "top_of_atmos_upward_shortwave_flux"],
    "toa_down_sw_radiative_flux": ["DSWRFtoa", "SOLIN"],
    "air_temperature": ["air_temperature_", "T_"],
}


# ---------------------------------------------------------------------------
# Packed-tensor corrector operations (self-contained, JIT-compatible)
# ---------------------------------------------------------------------------


def area_weighted_mean(
    field: Tensor, area_weights: Tensor, keepdim: bool = False
) -> Tensor:
    """Area-weighted mean over the last two (lat, lon) dims."""
    return (field * area_weights).sum(dim=(-2, -1), keepdim=keepdim)


def vertical_integral(
    integrand: Tensor,
    surface_pressure: Tensor,
    ak: Tensor,
    bk: Tensor,
    gravity: float,
) -> Tensor:
    """Mass-weighted vertical integral  (1/g) * integral(x dp).

    Args:
        integrand: [B, H, W, K]
        surface_pressure: [B, H, W]
        ak, bk: [K+1]
    """
    interface_p = ak + bk * surface_pressure.unsqueeze(-1)  # [B,H,W,K+1]
    dp = interface_p.diff(dim=-1)  # [B,H,W,K]
    return (integrand * dp).sum(dim=-1) / gravity


def force_positive_channels(output: Tensor, indices: Tensor) -> Tensor:
    output = output.clone()
    for i in range(indices.shape[0]):
        idx = indices[i].item()
        output[:, idx] = torch.clamp(output[:, idx], min=0.0)
    return output


def _stack_levels(packed: Tensor, indices: Tensor) -> Tensor:
    return torch.stack(
        [packed[:, indices[k].item()] for k in range(indices.shape[0])], dim=-1
    )


def conserve_dry_air_packed(
    input_packed: Tensor,
    output_packed: Tensor,
    area_weights: Tensor,
    ak: Tensor,
    bk: Tensor,
    gravity: float,
    ps_out_idx: int,
    water_out_indices: Tensor,
    ps_in_idx: int,
    water_in_indices: Tensor,
    precision: torch.dtype = torch.float64,
) -> Tensor:
    """Conserve dry air by correcting surface pressure.

    Matches the original corrector's precision path: TWP and dry-air pressure
    are computed in the input's native dtype (float32), then promoted to
    ``precision`` (float64) for the global-mean correction.
    """
    output_packed = output_packed.clone()

    def _dry_air_ps(packed: Tensor, ps_idx: int, wat_indices: Tensor) -> Tensor:
        # Compute TWP and dry-air in native precision (float32), matching
        # the original AtmosphereData.surface_pressure_due_to_dry_air which
        # calls vertical_coordinate.vertical_integral in float32 before
        # promoting to the corrector's precision.
        ps = packed[:, ps_idx]
        wat = _stack_levels(packed, wat_indices)
        twp = vertical_integral(wat, ps, ak, bk, gravity)
        return (ps - gravity * twp).to(precision)

    gen_dry = _dry_air_ps(output_packed, ps_out_idx, water_out_indices)
    inp_dry = _dry_air_ps(input_packed, ps_in_idx, water_in_indices)

    aw = area_weights.to(precision)
    error = area_weighted_mean(gen_dry, aw, True) - area_weighted_mean(
        inp_dry, aw, True
    )
    new_dry = gen_dry - error

    wat = _stack_levels(output_packed, water_out_indices).to(precision)
    ak_d = ak.to(precision).diff()
    bk_d = bk.to(precision).diff()
    new_ps = (new_dry + (ak_d * wat).sum(-1)) / (1 - (bk_d * wat).sum(-1))
    output_packed[:, ps_out_idx] = new_ps.to(output_packed.dtype)
    return output_packed


def zero_global_mean_moisture_advection_packed(
    output_packed: Tensor,
    area_weights: Tensor,
    advection_idx: int,
) -> Tensor:
    output_packed = output_packed.clone()
    adv = output_packed[:, advection_idx]
    mean_adv = area_weighted_mean(adv, area_weights, keepdim=False)
    output_packed[:, advection_idx] = adv - mean_adv[:, None, None]
    return output_packed


def moisture_correction_packed(
    input_packed: Tensor,
    output_packed: Tensor,
    area_weights: Tensor,
    ak: Tensor,
    bk: Tensor,
    gravity: float,
    timestep_seconds: float,
    terms_to_modify: str,
    ps_in_idx: int,
    water_in_indices: Tensor,
    ps_out_idx: int,
    water_out_indices: Tensor,
    evap_idx: int,
    precip_idx: int,
    advection_idx: int,
) -> Tensor:
    """Enforce moisture budget closure.

    The evap channel (evap_idx) stores latent heat flux in W/m^2.
    The original corrector converts to evaporation rate (kg/m^2/s) via
    LATENT_HEAT_OF_VAPORIZATION before computing the budget, then converts
    back when writing.  We must do the same here.
    """
    output_packed = output_packed.clone()

    def _twp(packed: Tensor, ps_idx: int, wat_idx: Tensor) -> Tensor:
        ps = packed[:, ps_idx]
        wat = _stack_levels(packed, wat_idx)
        return vertical_integral(wat, ps, ak, bk, gravity)

    gen_twp = _twp(output_packed, ps_out_idx, water_out_indices)
    inp_twp = _twp(input_packed, ps_in_idx, water_in_indices)

    tendency = (gen_twp - inp_twp) / timestep_seconds
    tendency_gm = area_weighted_mean(tendency, area_weights, keepdim=True)

    # Convert latent heat flux (W/m^2) → evaporation rate (kg/m^2/s)
    evap_rate = output_packed[:, evap_idx] / LATENT_HEAT_OF_VAPORIZATION
    evap_gm = area_weighted_mean(evap_rate, area_weights, keepdim=True)
    precip_gm = area_weighted_mean(
        output_packed[:, precip_idx], area_weights, keepdim=True
    )

    if terms_to_modify.endswith("precipitation"):
        scale = (evap_gm - tendency_gm) / precip_gm
        output_packed[:, precip_idx] = (
            output_packed[:, precip_idx]
            * scale.squeeze(-1).squeeze(-1).unsqueeze(1)
            .expand_as(output_packed[:, precip_idx : precip_idx + 1])
            .squeeze(1)
        )
    elif terms_to_modify.endswith("evaporation"):
        scale = (tendency_gm + precip_gm) / evap_gm
        # Scale the evaporation rate, then convert back to LHFLX
        output_packed[:, evap_idx] = (
            output_packed[:, evap_idx]
            * scale.squeeze(-1).squeeze(-1).unsqueeze(1)
            .expand_as(output_packed[:, evap_idx : evap_idx + 1])
            .squeeze(1)
        )
        # Update evap_rate for advection computation below
        evap_rate = output_packed[:, evap_idx] / LATENT_HEAT_OF_VAPORIZATION

    if terms_to_modify.startswith("advection"):
        output_packed[:, advection_idx] = tendency - (
            evap_rate - output_packed[:, precip_idx]
        )

    return output_packed


# ---------------------------------------------------------------------------
# Channel-index helpers
# ---------------------------------------------------------------------------


def _find_index_by_prefixes(names: list[str], prefixes: list[str]) -> int:
    for prefix in prefixes:
        for i, name in enumerate(names):
            if name == prefix or name.startswith(prefix):
                return i
    return -1


def _find_all_indices_by_prefix(names: list[str], prefix: str) -> list[int]:
    return sorted(i for i, n in enumerate(names) if n.startswith(prefix))


def _find_by_standard(names: list[str], std_name: str) -> int:
    prefixes = ATMOSPHERE_FIELD_NAME_PREFIXES.get(std_name, [])
    return _find_index_by_prefixes(names, prefixes)


def _find_all_by_standard(names: list[str], std_name: str) -> list[int]:
    prefixes = ATMOSPHERE_FIELD_NAME_PREFIXES.get(std_name, [])
    indices: list[int] = []
    for prefix in prefixes:
        indices.extend(_find_all_indices_by_prefix(names, prefix))
    return sorted(set(indices))


# ---------------------------------------------------------------------------
# Traceable module — the full ACE single-step
# ---------------------------------------------------------------------------


class TraceableACEModule(nn.Module):
    """Self-contained ACE single-step: normalize -> NN -> residual ->
    secondary decoder -> denormalize -> correctors -> ocean.

    forward(inputs) -> output

    ``inputs`` is a single tensor with shape [B, C_in + C_forcing, H, W].
    The first C_in channels are the prognostic/diagnostic state; the
    remaining C_forcing channels are the next-step forcing.
    """

    def __init__(
        self,
        module: nn.Module,
        secondary_decoder_module: nn.Module | None,
        in_names: list[str],
        out_names: list[str],
        all_out_names: list[str],
        forcing_names: list[str],
        in_means: Tensor,
        in_stds: Tensor,
        out_means: Tensor,
        out_stds: Tensor,
        residual_prediction: bool,
        prognostic_input_indices: Tensor,
        prognostic_output_indices: Tensor,
        force_positive_indices: Tensor,
        corrector_flags: dict[str, Any],
        area_weights: Tensor | None,
        ak: Tensor | None,
        bk: Tensor | None,
        timestep_seconds: float,
        corrector_channel_map: dict[str, Any],
        ocean_config: dict[str, Any] | None = None,
        include_normalization: bool = True,
    ):
        super().__init__()
        self.module = module
        self.secondary_decoder_module = secondary_decoder_module

        self.in_names = in_names
        self.out_names = out_names
        self.all_out_names = all_out_names
        self.forcing_names = forcing_names

        self.include_normalization = include_normalization
        self.register_buffer("in_means", in_means.view(1, -1, 1, 1))
        self.register_buffer("in_stds", in_stds.view(1, -1, 1, 1))
        self.register_buffer("out_means", out_means.view(1, -1, 1, 1))
        self.register_buffer("out_stds", out_stds.view(1, -1, 1, 1))

        self.residual_prediction = residual_prediction
        self.register_buffer("prog_in_idx", prognostic_input_indices)
        self.register_buffer("prog_out_idx", prognostic_output_indices)
        self.register_buffer("force_pos_idx", force_positive_indices)

        self.n_in = in_means.shape[0]
        self.corrector_flags = corrector_flags
        self.timestep_seconds = timestep_seconds
        self.corrector_channel_map = corrector_channel_map

        # Ocean SST prescription config
        self.ocean_config = ocean_config if ocean_config is not None else {}

        if area_weights is not None:
            self.register_buffer("area_weights", area_weights)
        else:
            self.area_weights: Tensor | None = None  # type: ignore[assignment]
        if ak is not None:
            self.register_buffer("ak", ak)
        else:
            self.ak: Tensor | None = None  # type: ignore[assignment]
        if bk is not None:
            self.register_buffer("bk", bk)
        else:
            self.bk: Tensor | None = None  # type: ignore[assignment]

    def forward(self, inputs: Tensor) -> Tensor:
        """One ACE timestep.

        Args:
            inputs: [B, C_in + C_forcing, H, W] concatenated state + forcing.
                When normalization is included (default), state channels are in
                physical (denormalized) units.  When normalization is excluded,
                state channels must be pre-normalized and forcing is omitted.
        Returns:
            [B, C_out, H, W] output tensor.
        """
        input = inputs[:, : self.n_in]
        forcing = inputs[:, self.n_in :]

        # 1) Normalize (or pass through if caller provides pre-normalized data)
        if self.include_normalization:
            x_norm = (input - self.in_means) / self.in_stds
        else:
            x_norm = input

        # 2) NN forward
        y_norm = self.module(x_norm)

        # 3) Residual
        if self.residual_prediction:
            for j in range(self.prog_out_idx.shape[0]):
                oi = self.prog_out_idx[j].item()
                ii = self.prog_in_idx[j].item()
                y_norm[:, oi] = y_norm[:, oi] + x_norm[:, ii]

        # 4) Secondary decoder
        if self.secondary_decoder_module is not None:
            sec_out = self.secondary_decoder_module(y_norm)
            y_norm = torch.cat([y_norm, sec_out], dim=1)

        # 5) Denormalize (or return normalized output)
        if self.include_normalization:
            output = y_norm * self.out_stds + self.out_means
        else:
            return y_norm

        # 6) Corrections
        output = self._apply_corrections(input, output, forcing)

        # 7) Ocean SST prescription
        output = self._apply_ocean(output, forcing)
        return output

    def _apply_ocean(self, output: Tensor, forcing: Tensor) -> Tensor:
        """Prescribe sea surface temperature over ocean grid points."""
        oc = self.ocean_config
        if not oc.get("active", False):
            return output

        ocnfrac_forcing_idx: int = oc["ocnfrac_forcing_idx"]
        target_ts_forcing_idx: int = oc["target_ts_forcing_idx"]
        ts_out_idx: int = oc["ts_out_idx"]
        interpolate: bool = oc.get("interpolate", False)

        ocean_frac = forcing[:, ocnfrac_forcing_idx]  # [B, H, W]
        target_ts = forcing[:, target_ts_forcing_idx]  # [B, H, W]

        output = output.clone()
        if interpolate:
            output[:, ts_out_idx] = (
                ocean_frac * target_ts + (1.0 - ocean_frac) * output[:, ts_out_idx]
            )
        else:
            mask = torch.round(ocean_frac).to(torch.int32)
            output[:, ts_out_idx] = torch.where(
                mask == 1, target_ts, output[:, ts_out_idx]
            )
        return output

    def _apply_corrections(
        self, input: Tensor, output: Tensor, forcing: Tensor
    ) -> Tensor:
        flags = self.corrector_flags
        cmap = self.corrector_channel_map

        if self.force_pos_idx.numel() > 0:
            output = force_positive_channels(output, self.force_pos_idx)

        if not flags.get("any_active", False):
            return output

        assert self.area_weights is not None

        if flags.get("conserve_dry_air", False):
            assert self.ak is not None and self.bk is not None
            output = conserve_dry_air_packed(
                input_packed=input,
                output_packed=output,
                area_weights=self.area_weights,
                ak=self.ak,
                bk=self.bk,
                gravity=GRAVITY,
                ps_out_idx=cmap["ps_out"],
                water_out_indices=torch.tensor(
                    cmap["water_out_indices"], device=output.device
                ),
                ps_in_idx=cmap["ps_in"],
                water_in_indices=torch.tensor(
                    cmap["water_in_indices"], device=output.device
                ),
            )

        if flags.get("zero_global_mean_moisture_advection", False):
            output = zero_global_mean_moisture_advection_packed(
                output_packed=output,
                area_weights=self.area_weights,
                advection_idx=cmap["advection_out"],
            )

        if flags.get("moisture_budget_correction") is not None:
            assert self.ak is not None and self.bk is not None
            output = moisture_correction_packed(
                input_packed=input,
                output_packed=output,
                area_weights=self.area_weights,
                ak=self.ak,
                bk=self.bk,
                gravity=GRAVITY,
                timestep_seconds=self.timestep_seconds,
                terms_to_modify=flags["moisture_budget_correction"],
                ps_in_idx=cmap["ps_in"],
                water_in_indices=torch.tensor(
                    cmap["water_in_indices"], device=output.device
                ),
                ps_out_idx=cmap["ps_out"],
                water_out_indices=torch.tensor(
                    cmap["water_out_indices"], device=output.device
                ),
                evap_idx=cmap["evap_out"],
                precip_idx=cmap["precip_out"],
                advection_idx=cmap["advection_out"],
            )

        if flags.get("total_energy_budget_correction", False):
            logger.warning(
                "Total energy budget correction is not yet implemented "
                "in the traced model.  Skipping."
            )

        return output


# ---------------------------------------------------------------------------
# Build from checkpoint
# ---------------------------------------------------------------------------


def _unwrap_step(stepper: Any) -> Any:
    step = stepper._step_obj
    if hasattr(step, "_wrapped_step"):
        step = step._wrapped_step
    return step


def _build_corrector_flags_and_channel_map(
    stepper: Any,
    in_names: list[str],
    all_out_names: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve corrector config into flags + integer channel map.

    Uses the local ATMOSPHERE_FIELD_NAME_PREFIXES copy so there is no
    runtime dependency on fme.core.atmosphere_data.
    """
    corrector = _unwrap_step(stepper)._corrector
    config = getattr(corrector, "_config", None)

    flags: dict[str, Any] = {"any_active": False}
    cmap: dict[str, Any] = {}

    # Check if it's an AtmosphereCorrectorConfig without importing the class
    # (attribute duck-typing keeps us loosely coupled).
    if config is None or not hasattr(config, "conserve_dry_air"):
        return flags, cmap

    flags["conserve_dry_air"] = config.conserve_dry_air
    flags["zero_global_mean_moisture_advection"] = (
        config.zero_global_mean_moisture_advection
    )
    flags["moisture_budget_correction"] = config.moisture_budget_correction
    flags["total_energy_budget_correction"] = (
        config.total_energy_budget_correction is not None
    )
    flags["any_active"] = any(
        [
            config.conserve_dry_air,
            config.zero_global_mean_moisture_advection,
            config.moisture_budget_correction is not None,
            config.total_energy_budget_correction is not None,
        ]
    )

    cmap["ps_in"] = _find_by_standard(in_names, "surface_pressure")
    cmap["ps_out"] = _find_by_standard(all_out_names, "surface_pressure")
    cmap["water_in_indices"] = _find_all_by_standard(
        in_names, "specific_total_water"
    )
    cmap["water_out_indices"] = _find_all_by_standard(
        all_out_names, "specific_total_water"
    )
    cmap["advection_out"] = _find_by_standard(
        all_out_names, "tendency_of_total_water_path_due_to_advection"
    )
    cmap["precip_out"] = _find_by_standard(all_out_names, "precipitation_rate")
    cmap["evap_out"] = _find_by_standard(all_out_names, "latent_heat_flux")

    return flags, cmap


def load_and_build(
    checkpoint_path: str,
    device: str = "cpu",
    include_normalization: bool = True,
    include_corrector: bool = True,
    include_ocean: bool = True,
) -> tuple[TraceableACEModule, dict[str, Any]]:
    """Load an ACE checkpoint and build a TraceableACEModule.

    The only import from fme is ``load_stepper`` — everything else is local.

    Args:
        checkpoint_path: Path to .ckpt or .tar checkpoint file.
        device: Target device for the traced model.
        include_normalization: If True (default), the model expects denormalized
            (physical-unit) inputs and produces denormalized outputs.  If False,
            inputs must be pre-normalized and output is normalized; this also
            forces include_corrector=False and include_ocean=False.
        include_corrector: If True (default), include atmosphere correctors
            (force-positive, dry-air conservation, moisture budget).
        include_ocean: If True (default), include ocean SST prescription
            when the checkpoint has an ocean model configured.
    """
    if not include_normalization:
        include_corrector = False
        include_ocean = False
    from fme.ace.stepper.single_module import load_stepper
    from fme.core.device import force_cpu

    logger.info("Loading checkpoint: %s", checkpoint_path)
    with force_cpu(device == "cpu"):
        stepper = load_stepper(checkpoint_path)
    step = _unwrap_step(stepper)
    config = step.config
    normalizer = step.normalizer

    in_names: list[str] = list(config.in_names)
    out_names: list[str] = list(config.out_names)

    # Secondary decoder
    sec_names: list[str] = []
    sec_module: nn.Module | None = None
    if config.secondary_decoder is not None:
        sec_names = list(config.secondary_decoder.secondary_diagnostic_names)
    all_out_names = out_names + sec_names

    sd = step.secondary_decoder
    if hasattr(sd, "_module"):
        sec_module = sd._module.torch_module

    forcing_names: list[str] = list(config.next_step_forcing_names)

    # Normalization stats
    in_means = torch.stack(
        [normalizer.means[n].float().squeeze() for n in in_names]
    )
    in_stds = torch.stack(
        [normalizer.stds[n].float().squeeze() for n in in_names]
    )
    out_means_l, out_stds_l = [], []
    for n in all_out_names:
        if n in normalizer.means:
            out_means_l.append(normalizer.means[n].float().squeeze())
            out_stds_l.append(normalizer.stds[n].float().squeeze())
        else:
            out_means_l.append(torch.tensor(0.0))
            out_stds_l.append(torch.tensor(1.0))
    out_means = torch.stack(out_means_l)
    out_stds = torch.stack(out_stds_l)

    # Prognostic indices
    prog_names = list(config.prognostic_names)
    prog_out_idx = torch.tensor(
        [all_out_names.index(n) for n in prog_names if n in all_out_names],
        dtype=torch.long,
    )
    prog_in_idx = torch.tensor(
        [in_names.index(n) for n in prog_names if n in in_names],
        dtype=torch.long,
    )

    # Force-positive indices
    corrector = step._corrector
    corr_config = getattr(corrector, "_config", None)
    fp_names: list[str] = []
    if corr_config is not None and hasattr(corr_config, "force_positive_names"):
        fp_names = corr_config.force_positive_names
    force_pos_idx = torch.tensor(
        [all_out_names.index(n) for n in fp_names if n in all_out_names],
        dtype=torch.long,
    )

    # Corrector flags
    if include_corrector:
        corrector_flags, corrector_cmap = _build_corrector_flags_and_channel_map(
            stepper, in_names, all_out_names
        )
    else:
        corrector_flags = {"any_active": False}
        corrector_cmap = {}

    # Area weights, ak, bk
    area_weights: Tensor | None = None
    ak_t: Tensor | None = None
    bk_t: Tensor | None = None
    timestep_seconds = stepper._dataset_info.timestep.total_seconds()

    if corrector_flags.get("any_active", False):
        gridded_ops = stepper._dataset_info.gridded_operations
        if hasattr(gridded_ops, "_cpu_area_global"):
            area_weights = gridded_ops._cpu_area_global.float()
        vc = stepper._dataset_info.atmosphere_vertical_coordinate
        if vc is not None and hasattr(vc, "ak"):
            ak_t = vc.ak.float()
            bk_t = vc.bk.float()

    # Ocean SST prescription — add forcing channels for mask and target SST
    ocean_cfg: dict[str, Any] | None = None
    ocean_forcing_names: list[str] = []
    if include_ocean and step.ocean is not None:
        ocean_obj = step.ocean
        sst_name = ocean_obj.surface_temperature_name
        ocnfrac_name = ocean_obj.ocean_fraction_name
        is_prescribed = ocean_obj.type == "prescribed"

        if not is_prescribed:
            logger.warning(
                "Slab ocean model is not yet supported in the traced model. "
                "Ocean SST prescription will be skipped."
            )
        else:
            ts_out_idx = (
                all_out_names.index(sst_name) if sst_name in all_out_names else -1
            )
            if ts_out_idx < 0:
                logger.warning(
                    "Ocean surface_temperature_name '%s' not in output channels. "
                    "Ocean SST prescription will be skipped.",
                    sst_name,
                )
            else:
                # Add ocean forcing channels right after the existing forcing_names
                ocean_forcing_names = [ocnfrac_name, sst_name]
                ocnfrac_forcing_idx = len(forcing_names)
                target_ts_forcing_idx = len(forcing_names) + 1
                ocean_cfg = {
                    "active": True,
                    "ocnfrac_forcing_idx": ocnfrac_forcing_idx,
                    "target_ts_forcing_idx": target_ts_forcing_idx,
                    "ts_out_idx": ts_out_idx,
                    "interpolate": ocean_obj.prescriber.interpolate,
                    "ocean_fraction_name": ocnfrac_name,
                    "surface_temperature_name": sst_name,
                }
                logger.info(
                    "Ocean SST prescription enabled: %s (mask=%s, interpolate=%s)",
                    sst_name,
                    ocnfrac_name,
                    ocean_obj.prescriber.interpolate,
                )

    # Combine regular forcing + ocean forcing for the full forcing channel list
    all_forcing_names = forcing_names + ocean_forcing_names

    raw_module = step.module.torch_module

    traceable = TraceableACEModule(
        module=raw_module,
        secondary_decoder_module=sec_module,
        in_names=in_names,
        out_names=out_names,
        all_out_names=all_out_names,
        forcing_names=all_forcing_names,
        in_means=in_means,
        in_stds=in_stds,
        out_means=out_means,
        out_stds=out_stds,
        residual_prediction=config.residual_prediction,
        prognostic_input_indices=prog_in_idx,
        prognostic_output_indices=prog_out_idx,
        force_positive_indices=force_pos_idx,
        corrector_flags=corrector_flags,
        area_weights=area_weights,
        ak=ak_t,
        bk=bk_t,
        timestep_seconds=timestep_seconds,
        corrector_channel_map=corrector_cmap,
        ocean_config=ocean_cfg,
        include_normalization=include_normalization,
    )
    traceable = traceable.to(device).eval()

    n_lat, n_lon = stepper._dataset_info.img_shape
    metadata = {
        "input_channels": {
            **{i: name for i, name in enumerate(in_names)},
            **{
                i + len(in_names): f"{name}_next_step"
                for i, name in enumerate(all_forcing_names)
            },
        },
        "output_channels": {i: name for i, name in enumerate(all_out_names)},
        "n_lat": n_lat,
        "n_lon": n_lon,
        "input_shape": f"[batch, {len(in_names) + len(all_forcing_names)}, {n_lat}, {n_lon}]",
        "output_shape": f"[batch, {len(all_out_names)}, {n_lat}, {n_lon}]",
        "n_input_channels": len(in_names),
        "n_forcing_channels": len(all_forcing_names),
        "residual_prediction": config.residual_prediction,
        "normalization_enabled": include_normalization,
        "corrector_enabled": include_corrector,
        "corrector_flags": corrector_flags,
        "timestep_seconds": timestep_seconds,
        "ocean_enabled": ocean_cfg is not None,
        "ocean_config": ocean_cfg,
    }
    return traceable, metadata


# ---------------------------------------------------------------------------
# Trace and save
# ---------------------------------------------------------------------------


def trace_and_save(
    checkpoint_path: str,
    output_base: str = "ace_traced",
    device: str = "cpu",
    include_normalization: bool = True,
    include_corrector: bool = True,
    include_ocean: bool = True,
) -> pathlib.Path:
    traceable, metadata = load_and_build(
        checkpoint_path,
        device=device,
        include_normalization=include_normalization,
        include_corrector=include_corrector,
        include_ocean=include_ocean,
    )

    n_in = metadata["n_input_channels"]
    n_forcing = metadata["n_forcing_channels"]
    n_out = len(metadata["output_channels"])
    n_lat = metadata["n_lat"]
    n_lon = metadata["n_lon"]

    n_total_in = n_in + n_forcing if include_normalization else n_in
    example_inputs = torch.randn(1, n_total_in, n_lat, n_lon, device=device)

    layers = []
    if include_normalization:
        layers.append("normalize")
    layers.append("NN")
    if include_corrector:
        layers.append("correctors")
    if include_ocean and metadata.get("ocean_enabled", False):
        layers.append("ocean")
    logger.info(
        "Tracing model [%s] (%d in, %d out, %d forcing, %dx%d grid) ...",
        " → ".join(layers),
        n_in,
        n_out,
        n_forcing,
        n_lat,
        n_lon,
    )
    with torch.no_grad():
        traced = torch.jit.trace(traceable, (example_inputs,))

    out = pathlib.Path(output_base + "_" + device)
    pt_path = out.with_suffix(".pt")
    meta_path = out.parent / (out.stem + "_metadata.yaml")

    torch.jit.save(traced, str(pt_path))
    logger.info("Saved TorchScript model: %s", pt_path)

    with open(meta_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    logger.info("Saved metadata: %s", meta_path)

    return pt_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    from fme.core.distributed import Distributed

    parser = argparse.ArgumentParser(
        description="Trace an ACE model for single-step inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
pipeline layers (additive — each one requires the previous):

  (bare)                Raw NN only.  Input must be pre-normalized;
                        output is normalized.
                        Input: [B, C_in, H, W]

  --add-normalization   Wrap the NN with input normalization and output
                        denormalization.  Input/output in physical units.
                        Input: [B, C_in + C_forcing, H, W]

  --add-corrector       Add atmosphere correctors on top of normalization
                        (force-positive, dry-air conservation, moisture
                        budget).  Requires --add-normalization.
                        Input: [B, C_in + C_forcing, H, W]

  --add-ocean           Add ocean SST prescription on top of correctors.
                        Requires --add-normalization.
                        Input: [B, C_in + C_forcing, H, W]

  --all                 Shorthand for --add-normalization --add-corrector
                        --add-ocean (full production pipeline).

examples:
  # Full production pipeline on GPU
  %(prog)s checkpoint.ckpt --all --device cuda

  # NN + normalization only (no physics corrections)
  %(prog)s checkpoint.ckpt --add-normalization

  # NN + normalization + correctors (no ocean)
  %(prog)s checkpoint.ckpt --add-normalization --add-corrector

  # Raw NN (normalized I/O, smallest model)
  %(prog)s checkpoint.ckpt
""",
    )
    parser.add_argument("checkpoint", help="Path to .ckpt or .tar checkpoint file")
    parser.add_argument(
        "output",
        nargs="?",
        default="ace_traced",
        help="Output base name (default: ace_traced)",
    )
    parser.add_argument("--device", default="cpu", help="Device (default: cpu)")
    parser.add_argument(
        "--add-normalization",
        action="store_true",
        help="Include input normalization and output denormalization.",
    )
    parser.add_argument(
        "--add-corrector",
        action="store_true",
        help="Include atmosphere correctors (force-positive, dry air, moisture). "
        "Requires --add-normalization.",
    )
    parser.add_argument(
        "--add-ocean",
        action="store_true",
        help="Include ocean SST prescription. Requires --add-normalization.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Enable all pipeline layers (normalization + correctors + ocean).",
    )
    args = parser.parse_args()

    if args.all:
        args.add_normalization = True
        args.add_corrector = True
        args.add_ocean = True

    if (args.add_corrector or args.add_ocean) and not args.add_normalization:
        parser.error(
            "--add-corrector and --add-ocean require --add-normalization "
            "(correctors and ocean operate on denormalized physical units)."
        )

    with Distributed.context():
        trace_and_save(
            args.checkpoint,
            args.output,
            args.device,
            include_normalization=args.add_normalization,
            include_corrector=args.add_corrector,
            include_ocean=args.add_ocean,
        )


if __name__ == "__main__":
    main()
