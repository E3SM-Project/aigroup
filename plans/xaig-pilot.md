# xaig plan: a pilot, then a feature study, then a small circuit

**Status:** proposal for group review. Nothing here authorizes package changes, data movement or model runs.
**Updated:** October 3, 2026, after three rounds of review: an independent multi-lens review of an earlier draft, a reconciliation of that review against the pinned sources, and two reviews of this revision (one of which tightened the decision logic in §6.3, §7.4 and §8). Earlier drafts and the review record are kept with the authors, outside the repository. Part I is what the group acts on now; Part II is reference material.

**Pinned sources**

| What | Pin | Status |
|---|---|---|
| xaig | v0.5.0, [`7e0e757`](https://github.com/E3SM-Project/aigroup/commit/7e0e757515a2e49724e8c134b926a49227c6db8d) | Source read; suite passed locally on Python 3.14 only (Part II §G) |
| E3SM ACE fork | [`e273d543`](https://github.com/E3SM-Project/ace/commit/e273d543892e6e1da38e1bf2aa252ba15deb0f0f) | Source read; no checkpoint run |
| Exporter branch | `origin/indu/sae` at [`f955ea35`](https://github.com/E3SM-Project/aigroup/commit/f955ea353f3ba23624c20da274716a734cb8a623) | Source read; unmerged, predates the 0.4.0 rename |
| Checkpoint configs | `archive/*.yaml` in aigroup | Configs show intended runs, not the bytes in a checkpoint; embedded configs, normalization files and data membership are unverified |

---

# Part I — what the group acts on now

## 1. The question, the claims and the milestones

**Question.** Do sparse features give a useful, reproducible account of what an ACE emulator computes — compact physical descriptions, accessible predictions and controlled behavioral effects — better than matched alternatives? Activations are deterministic functions of the inputs and the noise, so a feature cannot carry information absent from them. What a feature can offer is organization: computation exposed more clearly, cheaply or faithfully than by channels, PCA, probes or the inputs themselves.

**Contribution.** MacMillan & Ouellette (2025) already report SAE features in GraphCast aligned with atmospheric rivers and tropical cyclones, so "we found an AR feature" is a replication. Candidate contributions, to be tested rather than assumed: a causally validated feature under AMIP forcing with prespecified controls; transfer across the group's independently trained checkpoints; and later a bridge to E3SM itself. **Decision to record before the pilot:** AR-first (default) or cross-checkpoint universality first, with the reason.

**Phenomenon.** Atmospheric rivers (AR) are the provisional first target: synoptic scale, frequent, resolved at 1°, and built from fields ACE carries (`specific_total_water_k`, `U_k`, `V_k`, `PS`). The repo's own task list names a tropical-cyclone probe; at 1° TC intensity is under-resolved, so AR stays first unless label feasibility (§4) fails.

**Five claims, each with its own evidence.** (1) *Representation*: a feature reproducibly describes activations. (2) *Encoding*: the physical structure the label names causes the feature to fire (input-side tests). (3) *Model mechanism*: an intervention in the emulator changes its output as predicted. Claim (3) has two parts, tested separately (§8): a *causal-effect* claim (the patch changes the prespecified endpoint) and an *SAE-advantage* claim (the SAE direction does so better than matched alternatives at equal perturbation cost and collateral damage). (4) *Physical interpretation*: the feature and its effects agree with independent physical diagnostics. (5) *Physical causation*: the same perturbation in E3SM agrees. Correlation and reconstruction establish none of (2)–(5). Each record states which claim a number supports.

**Milestones.**

- **Pilot** — one checkpoint, one middle SFNO site, bounded export, TopK and PCA, strong baselines, lead-1 interventions. Ends with a frozen report and one of: *proceed, narrow, redesign, stop* (§8).
- **Milestone A** — a reproducible real-model feature study: verified identities, disjoint discovery/validation/test samples with emulator-training membership recorded, a minimal atlas with positives and hard negatives, case-level uncertainty, and calibrated lead-1 interventions with controls. No circuit, donor patching or mediation.
- **Milestone B** — one small computational path whose predictions are tested prospectively by original-model interventions on held-out cases (Part II §A).

A null or negative result completes any stage if its design and evidence are complete. Everything else (crosscoders, larger circuits, causal abstraction, physical twins, B-spline and other extensions) is an unscheduled backlog with entry criteria (Part II §C).

## 2. Sequence and gates

This table defines every gate; later sections point here instead of restating them.

| Stage | Output | Gate (quantity) | Frozen where / when | Role |
|---|---|---|---|---|
| Safety | PR1, PR1b (§5) | Regressions for §3.5 defects 1–3 pass on all CI tiers. Defects 4–5 are gated with PR5; defect 6 is handled by reporting the rank p-value as descriptive | — | xaig maintainer |
| Pilot prep | Checkpoint, site, labels, custody, budgets, first contribution (§4) | Every row of §4 has an answer and an owner | Pilot decision record, started here | PI and the roles in §4 |
| Capture | PR2, PR3a, PR3b (§5) | Kill/resume test; noise identical across batch size, rank count and resume; a small export reloads with verified identities | — | Exporter co-owners |
| Calibration export | About one month of contiguous 6-hourly lead-1 latents at one site | Noise floors, replay floor (capture-only reruns, §7.4), precision, and measured peak memory and transfer only | Before thresholds and budgets are frozen, and before the bulk export | Run operator |
| Intervention tooling | PR4, the live ACE adapter, and the PR5 arms that §6.1 item 6 needs (§5) | T1–T3 pass; a zero-edit arm equals the control within the measured replay tolerance (bitwise only where deterministic execution is enforced and shown to rerun bitwise); per-arm `RandomState` restore test | — | xaig maintainer and run operator |
| Design freeze | Pilot decision record, part 1: the design quantities listed in §8 | Every design quantity has a value; the power calculation is recorded | After calibration, before the bulk export | PI and phenomenon specialist |
| Bulk export | Discovery, validation and test windows (§4 Sampling) at the pilot site, spanning at least two years or both members | Within the frozen budget; reloads with verified identities; split manifest recorded | After the design freeze | Run operator |
| Candidate freeze | Pilot decision record, part 2: basis hashes, candidate IDs, dose values and control covariances (§8) | All recorded and hashed | After discovery and validation fits, before any test access | Analyst, countersigned by the PI |
| Pilot measurements | §6.1, using the §7 design for the item-6 arms only | All pilot must-haves (§6.1) reported | Margins frozen a priori (§8) | Analyst |
| Decision | Frozen pilot report | Decision rule (§8) | Before any test look | PI and phenomenon specialist |
| Milestone A | §9 | Milestone A gate (§9) | Study spec hash in the record | Group |
| Milestone B | Part II §A | Prospective path prediction within a frozen tolerance | Study spec hash in the record | Group |

A tiny toy check (§7.4, T0), with its small learned-dictionary bridge to `toy-dynamics` as a PR beside PR1, runs alongside the safety work. It gates nothing in the pilot, but must pass before the Milestone A gate (§9). The eight synthetic benchmark families of the previous plan are backlog.

**Calendar.** Three to four weeks for the pilot is an estimate conditional on accessible checkpoints and data, a named run operator, a GPU allocation, and code PRs proceeding in parallel: week 1 safety, prep and labels; week 2 capture PRs, calibration export and fits; week 3 analysis API, adapter, runner and lead-1 interventions; week 4 report and decision (cross-year or cross-member replication belongs to Milestone A, §9). Record resource availability before setting dates, and re-estimate once PR3b merges.

## 3. Facts the design rests on

Verified in the pinned sources on October 3, 2026. Re-verify against each checkpoint's embedded config before launch.

### 3.1 Checkpoints and data

- All five configs in `archive/` train ACE2/ACE2S on **E3SM AMIP output** (members 0101 and 0151, EAMv3). Observational AR catalogs do not apply; labels must come from a detector run on E3SM fields.
- All five build **`NoiseConditionedSFNO`** (isotropic noise, `noise_embed_dim` 64, `embed_dim` 384, 8 blocks, `filter_type: linear`). The two `ace2` configs use an MSE loss; the three `ace2s` configs use `EnsembleLoss` (0.9 CRPS + 0.1 energy score, two ensemble members). **Both kinds draw noise**. Measure effective noise sensitivity; do not infer it from the loss.
- The 65-year configs have **no training subset**: every same-member state is training data. Their validation windows (1994–95 for 0101, 1996–2000 for 0151) lie inside the training data.
- The 10-year config lists ten training years but repeats 1995: **nine unique years**. Its validation window, 1994–95, overlaps training year 1995.
- **Both ACE2S member-0101 configs (10-year and 65-year)** open with a line naming an X-SHiELD-AMIP-1deg stats dataset mounted at `/statsdata`, and take their centering, scaling and time-mean files from it. Only `ace2s_amip0151` and the two `ace2` configs use member-specific E3SM stats. Confirm the normalization source of any ACE2S 0101 checkpoint before choosing it.
- **Checkpoint file matters.** In the 10-year config, `best_inference_ckpt` is chosen on inline inference run every epoch from 16 starts (1951–2009; about 1951–2014 if 6-hourly), so with that file nearly the whole record influenced selection. `best_ckpt` is chosen on validation loss (1994–95).
- Studying training-distribution states is legitimate: it is a claim about the emulator's computation. "Unseen weather" requires states outside the emulator's training data, for example the other member's states run through a fixed checkpoint. Shared AMIP SST forcing means those are not independent forcing regimes. Held-out years are **unknown until the data manifest, normalization source and checkpoint file are confirmed**.
- The existing real archive is SamudrACE-E3SMv3, a coupled preindustrial control. It is a separate follow-up; prescribed-SST reasoning does not apply to it.
- **Custody.** Training data, normalization files (`centering.nc`, `scaling-*.nc`, `time-mean.nc`) and experiment directories sit on personal Perlmutter `$PSCRATCH` paths (`/pscratch/sd/o/olawale/…`, `/pscratch/sd/i/imanick/…`) or on container mounts: `/traindata` for data, `/statsdata` (an external X-SHiELD stats dataset) for the ACE2S 0101 normalization files, and `/output` for their experiment directories. Scratch is purged, and a hash does not help once the bytes are gone.

### 3.2 The SFNO block

For these configs (`data_grid` at its Legendre–Gauss default, `filter_residual` off), every block runs on one grid and the outer skip is exactly the normalized input:

$$a=\mathrm{norm0}(x),\qquad x_{\rm out}=\mathrm{MLP}\big(\mathrm{norm1}(\mathrm{GELU}(K a + b_K + W a + b_W))\big)+a .$$

- `norm0`, `norm1`: per-node layer norm over channels, `eps` 1e-5, plus a **noise-conditioned** scale and bias (1×1 convolutions of the 64-channel noise field), in every block.
- `K`: linear spectral convolution (spherical harmonic transform (SHT), degree-indexed contraction, inverse SHT) plus a per-channel bias `b_K`; the block's only spatial mixing. `W`: pointwise inner skip with bias `b_W`. MLP: pointwise, 384 → 768 → 384, with biases.
- `block_out(l)` is `block_in(l+1)`. The update attributable to the MLP is `mlp_out`, not `block_out − block_in`.
- Edits at `block_in` reach the block only through `norm0`: a uniform channel shift vanishes, and scaling a node's whole vector does almost nothing.
- Indu's `node_norm` (on `indu/sae`) reproduces `norm0`'s deterministic part (same formula, `eps` 1e-5; agreement is limited by fp32). It excludes the noise affine. Capturing after `norm0` instead includes the noise affine. **Decide which, explicitly** (§4).
- The decoder is two 1×1 convolutions around an activation, applied to the last block's output concatenated with the big-skip input; it is not linear.

### 3.3 Noise and state

- Noise is drawn **once per forward call** and shared by every block's norms. `isotropic_noise` draws the real coefficients for the whole batch, then the imaginary ones, so **even sample 0's noise depends on batch size**. The exporter's comment "the same noise, however split" is false.
- `fme.core.rand.use_generator` routes draws through one CPU generator for the whole batch. `StepperState` carries `corrector_state` and `random_state`; the `RandomState` **advances in place**, so arms that reuse it draw different noise. An unseeded run draws from the device RNG, which `StepperState` does not carry.
- The neural step is Markov in the physical state. The deployed step also carries the dry-air target (seeded once from the initial condition) and the RNG state. **At lead 1 from a reference state the dry-air target equals the input's own, so it does not matter; the RNG state does**: every arm must start from the same restored generator state, or the first noise draw is not paired. Multi-step replay must also restore the dry-air target, keyed noise and forcing.

### 3.4 What happens after the network

Order: `ForcePositive` (15 fields clamped), then `ConserveDryAir`, then `MoistureBudgetCorrection` (`advection_and_precipitation`), then the ocean step.

- Global **dry-air mass** is pinned to the initial-condition value (up to fp32 rounding of PS). Global PS still moves with water loading.
- Precipitation is multiplied by one global ratio, (gmean E − gmean dTWP/dt) / gmean P, and advection is reset per column to close each column's budget, so **global-mean advection is zero by construction**. Global P, E and TWP can respond to an intervention. A local change rescales P everywhere: a non-network, non-local path. If the ratio is negative, P becomes negative wherever it was positive, and nothing clamps it again; a ratio of zero sets P to zero everywhere; if gmean P = 0, the ratio divides by zero.
- No energy correction is configured.
- The ocean step overwrites TS with the prescribed value wherever the next step's `OCNFRAC` rounds to 1 (OCNFRAC > 0.5; exactly 0.5 rounds to 0), with no interpolation, so global-mean TS is partly prescribed.

Consequences: primary endpoints are regional; global budget closure is a diagnostic, not an endpoint. For every arm the adapter reports raw-network, pre-corrector, post-corrector and post-ocean outputs, the per-step precipitation ratio (flagging ≤ 0), the dry-air offset, clamp counts, and which cells the ocean step (and any configured prescribed-prognostic overwrite) replaced, since the fork's correction delta omits them. Pre-corrector responses describe the network; corrected responses describe the deployed model.

### 3.5 Defects in shipped xaig (reproduced)

1. An equal-width transcoder passes `feature_direction`'s width check and steers the wrong layer; all four arms run. `feature_direction` runs before any arm, so a guard there also protects the reconstruction arm.
2. `evaluate_basis` accepts a `target_layer` that differs from the fitted one, and `fit_sae` records no target network position.
3. A basis fitted on `split.buffer` is accepted as `other`. The leak check compares bare time labels, so members sharing a calendar are confused.
4. One non-finite field in any arm aborts `run_steering`, and nothing is returned.
5. `run_steering` records the steered layer at every step of every arm. Peak memory is three rollouts, about 3 × steps × 99.5 MB per recorded site.
6. The random-direction rank p-value has a floor of 1/21 at the default 20 draws. Bonferroni, Holm and Benjamini–Yekutieli can never reject at that floor for two or more tests; Benjamini–Hochberg and Hochberg reject only if essentially the whole family sits there. Treat it as descriptive.

## 4. Pilot preparation: decisions and people

Resolve each before bulk export and record the answer in the pilot decision record.

| Decision | Default | What settles it |
|---|---|---|
| First contribution | AR-first | The §1 decision and its reason. If universality-first, the pilot stays single-checkpoint and AR-targeted, and Milestone A adds a second checkpoint with a shared site definition (Part II §C) |
| Checkpoint | An in-house ACE2S AMIP checkpoint (member and config length named here), because its training membership can be checked against the group's own data and the other member supplies unseen weather. Public ACE2-EAMv3 instead if reproducibility by outsiders outweighs that and its training period and held-out years check out | Training window, normalization source (§3.1: avoid an unconfirmed `/statsdata` source), checkpoint file (`best_ckpt` or `best_inference_ckpt`, §3.1), timestep; public availability decides who can reproduce the study |
| Unseen-weather population | In-house checkpoint: the other member's states through the fixed checkpoint, reported side by side with same-member results under the same year-season clustering. ACE2-EAMv3: years its manifest shows outside training and checkpoint selection | Verified membership and a custody copy |
| Site and normalization | One middle block: `block_in` with node normalization, or `norm0_out` | The §3.2 trade-off, informed by the chosen checkpoint's measured noise sensitivity and the noise-affine (`W_scale_2d`, `W_bias_2d`) magnitudes at the site: deterministic layer-norm coordinates (per-node edit mapping needed) or the exact coordinates the block reads (noise affine included) |
| Labels | One detector on E3SM fields for the pilot, stated as a limitation; two of different type for Milestone A | ACE-resolution integrated vapor transport (IVT), (1/g) Σ_k q_k V_k Δp_k with Δp from the hybrid coefficients and PS, validated on a subset against E3SM's native vertically integrated moisture fluxes (TUQ/TVQ). `specific_total_water` includes condensate, and layer means drop within-layer covariance. Freeze detector thresholds (monthly or basin) and output time-averaging conventions with the labels |
| Sampling | Contiguous 6-hourly windows around detected events, plus matched non-AR windows | The 29-step stride (7.25 days) suits discovery, not event structure; `--every 1` with start and stop already gives one dense window |
| Custody | Group-owned storage (CFS project space or HPSS) and a group Hugging Face dataset pinned by revision | Copy and checksum checkpoints, normalization files and pilot-year data before PR2's filesystem test |
| Budgets | The lead-1 envelope (§7.5) | Storage, GPU-hours, operator |

| Work | Accountable role (to be agreed) | Evidence before launch |
|---|---|---|
| Margins and pilot decision | PI, with the phenomenon specialist | Frozen decision record (§8) |
| Pilot analysis (PR4, §6.1) | Analyst | Analysis functions merged; split manifest |
| Safety PRs, PR5 and the live adapter | xaig maintainer, with the run operator for the adapter | PR1 and PR5 merged; T1–T2 pass |
| Exporter and writer port | Indu as co-owner, subject to their agreement, with the xaig maintainer | Hand-port with kill/resume and noise-invariance tests |
| Labels and IVT | Atmospheric scientist or data analyst | Native-vs-coarse IVT audit, detector version |
| Data custody | Dataset and checkpoint custodian | Group storage, manifests, checksums |
| GPU runs | Named run operator | Allocation, pinned Python 3.11 environment |
| Scientific review | Phenomenon specialist | Baselines, confounders, endpoints, margins |
| Physical twins (backlog) | E3SM experiment lead | Cost probe first (Part II §C) |

These are roles to fill, not assignments anyone has accepted. Missing custody or access blocks launch; safety work continues regardless.

## 5. Next pull requests

Each is about 1,000 lines or fewer, useful on its own, and leaves `main` working.

**PR1 — measurement safety.** `latents/steering.py`, `latents/evaluate.py`, `nn/train.py`, tests, docs.
- Before any arm runs, refuse any `Dictionary` with `output_mean` or `output_scale` set, or with `fitted_on.target_layer` different from the read layer; remove or make explicit the `output_scale` branch in `feature_direction`. Regression: an equal-width (4→4) transcoder, naming both the feature and reconstruction arms.
- `fit_sae` records the target's network position. `evaluate_basis` refuses a `target_layer` that disagrees with the fitted one (or requires an explicit override flag) and checks the target's position the way `check_basis_fits` checks the read layer.
- The leak check refuses, or distinctly flags, fits that overlap `split.buffer`, and compares the *sample population* — provenance source plus `LatentInfo.identity()` — rather than bare labels. Regression: two members with the same calendar labels.
- Legacy bases without target identity stay usable only through the existing explicit unverified path. Regression: an archive subset whose stored layer indices differ from network positions, for both the read and the target layer.
- Docs: a note in `docs/package/nn.md` (its ordered task list) and `docs/package/index.md` that the group is taking a pilot-first order (§10).
- Split PR1 if identity migration exceeds the review budget.

**PR1b (small) — elapsed-time buffers** in `split_time_blocks`, using `LatentInfo.elapsed_seconds()`.

**PR2 — streaming archive writer**, hand-ported from `indu/sae` part A (`start_archive`, `ArchiveFiller`, `finish_archive`) and co-authored with Indu. Data is fsynced before a cell's completion marker; each rank owns its markers. Tests: kill and resume, interrupted writes, overlapping ranks, refused finish of an incomplete archive, atomic manifest publication. Add an `nn` CI tier on Python 3.11 (CI runs torch only on 3.13 today). Validate on the production filesystem before launch. Keep `node_norm` and training-history changes out of this PR.

**PR3a — capture-only ACE exporter: port and identity.** Rebuild `ace_export` on current names (no `daig` or `taig`); record `network_layer` and site labels (`block_in`, `norm0_out`, `norm1_out`, `mlp_out`); persist checkpoint, config, normalization and data-manifest hashes and the sampling mode; one or two sites only. Decide here, once, where ACE code lives: in xaig behind an `ace` extra with lazy imports — which makes a network-free synthetic entry in `INTERVENABLE_CASES` (`tests/test_adapter_contracts.py`) mandatory for the live adapter — or exporter and live adapter together in one external plugin distribution with its own contract run. Default: one external plugin during the pilot. If §4 chooses `block_in` with node normalization, port `node_norm` here (co-authored with Indu) and persist its `eps` and per-node statistics; the live adapter implements the per-node edit mapping that T2 tests.

**PR3b — keyed noise and reference fields.** Batch size 1 per case, or per-case noise assembled into the batch, from `RandomState.from_seed(seed)`, where `seed` is a stable digest (for example the first 8 bytes of SHA-256 of a canonical `experiment/case/noise-member/step` string), never Python's per-process salted `hash()`. Record the key format, the digest and any disabled-noise setting in the archive manifest. Tests that noise is identical across batch size, rank count and resume. Write the reference fields the labels need (q, U, V and PS per layer; precipitation before and after correction). Add several event windows per archive only if one dense window per run proves insufficient.

**PR4 — pilot analysis API.** The §6.1 must-haves as Python functions that return records: held-out fidelity against PCA at matched sparsity (mean active features), the nuisance and random-init audits, the probe and IVT baselines; and the `pilot-decision` and `instrument-card` record kinds in `record.py`, validated and versioned. Until PR4 merges, the pilot decision record is a dated, hashed document.

**PR5 — study runner.** `run_steering` hard-codes control, reconstruction, feature and isotropic random arms and reports only global means, so the runner composes it or adds a pluggable arm and control-sampler parameter. The pilot needs only the item-6 pieces (§6.1): difference-in-means (DiffMean, §6.2), probe and covariance-aware random directions; regional endpoints; typed per-arm failures persisted as each arm completes. Milestone A adds the `h_RF` arm (§7.2), unrelated features, a placebo region and dose response. Also: opt-in latent recording (edit sites and selected steps only); streamed `latent_rms` and endpoint reductions; releasing each finished rollout before the next; chunking the one remaining full-width transform (`analysis.py:338`, all features over a region: about 2.1 GB at 4,096 float64 features over the globe); the `study-spec` record kind and the §11 pre-publication validator. Amend the finiteness rule in `latents/AGENTS.md` and the steering non-goals in the same PR (§10).

**Optional — training quality.** An auxiliary dead-feature loss (AuxK) and a learning-rate schedule, roughly 100–300 lines, if the pilot's held-out dead fraction exceeds the frozen tolerance (§8). Tied initialization already ships (`sae.py:121-126`); it is skipped when output and input widths differ.

The live intervention adapter — `Intervenable` for ACE, with keyed noise and a restored `RandomState` per arm, raw/pre/post-corrector outputs, and site hooks that copy GPU↔numpy while preserving padding — follows PR3b and precedes the pilot's lead-1 step. It needs no gradients.

## 6. Pilot measurements

### 6.1 Must-haves

The pilot reports these and nothing more; everything else in this plan belongs to Milestone A or later.

1. PR1, PR1b, PR2, PR3a, PR3b, PR4, the live adapter and the PR5 pieces item 6 needs merged; the calibration export reloads with verified identities.
2. TopK SAE and PCA at one middle site: held-out fraction of variance unexplained at matched mean active features, on a different year (or member) than fitting; held-out dead and rarely-firing fractions.
3. Nuisance audit: the share of alive features whose activation is predicted by static geography and clock (lat, lon, `LANDFRAC`, `OCNFRAC`, `SOLIN`, day of year, hour). Reported; not used to stop.
4. Random-init and input nulls: the same SAE fitted at the same site of a re-initialized SFNO, and on the stacked normalized inputs.
5. AR detection on IVT-matched cases: feature, dense probe on the 384 channels and IVT magnitude, scored as area under the precision–recall curve (AUPRC) within IVT deciles and against IVT-matched non-filamentary negatives.
6. Lead-1 interventions for at most three candidate features: feature, DiffMean, a validation-tuned probe direction, and norm- and covariance-matched random directions, with keyed noise, the measured replay floor, and calibration tiers T1–T3 (§7.4).

### 6.2 Baselines and what counts

- Dense regularized probes and a difference-in-means (DiffMean) direction, tuned on validation only, for detection *and* for steering.
- Local IVT, integrated water vapor (IWV) and wind speed; neighborhood or multiscale IVT; the seasonal and location climatology of AR frequency.
- SAEs on the stacked inputs and on a random-init SFNO, matched in width, sparsity, seeds and splits.

No feature is expected to beat the label-defining detector at reproducing its own labels from complete inputs. The test is **value beyond IVT magnitude at matched IVT**. A mid-network latent has passed through global spectral convolutions, so a per-node feature can legitimately beat local-input baselines on a geometric label. A feature that adds nothing beyond IVT magnitude is reported as an *IVT-magnitude feature*, not as an AR feature.

### 6.3 Samples, splits and uncertainty

- Identify samples by dataset, member, initialization and valid time, sampling mode, noise member and event. Keep emulator-training membership separate from dictionary-fitting membership. Network depth, forecast lead, valid time and rollout step are separate axes; independent checkpoints have independent coordinates, and equal widths do not identify spaces.
- Name the weighting: evaluation weights nodes by area and stored snapshots equally; report event-balanced or elapsed-time-weighted estimands separately.
- **Case sampling rule:** one case per event, at least 10–15 days from any other case in the same season and region; whole event windows stay in one partition. This is a sampling rule, not a demonstration of independence; independence is handled by clustering (below).
- Discovery fits dictionaries; validation selects features and hyperparameters; the test set is looked at once. Keep a dated deviation log and label every number exploratory or confirmatory.
- **Clustering:** the cluster is the year (or season within year), because AMIP cases in one year share prescribed SST. Cases from both members in the same year share forcing and belong to the **same** cluster: a second member adds cases, not clusters. Noise draws are nested within cases; nodes are not replicates. Report the number of clusters with every interval, and use a wild cluster bootstrap (cluster = year) rather than a pairs bootstrap.
- **Few clusters are not fixed by more resamples.** With a handful of years, cluster-robust intervals under-cover (Cameron, Gelbach & Miller 2008). The pilot is therefore exploratory whatever its size (§8).
- **Confirmatory status** (Milestone A) requires: a clustering design frozen at the design freeze; at least N_min year-clusters of held-out cases, with N_min set a priori (proposed default 20); and demonstrated interval coverage at that cluster count, both on zero-effect placebo patches (zero edits, placebo region) in the real model and on synthetic data with matching dependence. Evaluating a fixed in-house checkpoint on the other member's states makes this attainable: up to 65 years of weather it never saw.
- Confirmatory family: one signed primary endpoint, lead and dose per candidate and claim, with Holm across candidates (§8). Exploratory screening may use FDR under stated dependence. A larger random-direction budget needs a justified exchangeable null, not only a finer p-value.

## 7. Lead-1 intervention design

The pilot uses the $h_F$ contrast and the controls named in §6.1 item 6. The $h_{RF}$ arm, unrelated features, the placebo region and dose response are Milestone A (§9).

### 7.1 Why lead 1

Edit at forward step zero from independent reference states; the primary endpoint is that six-hour step's output. Discovery latents are also one step from reference states, so edits act on the distribution the dictionary was fitted on. At this lead the deployed state carries nothing beyond the reference input, and chaos, drift, memory and replay tolerance all stay small. Multi-step rollouts are a separately labeled follow-up ("propagation through the emulator state"), with lead-dependent replay floors.

### 7.2 Contrasts

For a same-space linear decoder, the feature arm is

$$h_F = h + \Delta\, d = \hat h(z') + [h - \hat h(z)].$$

- **Feature effect in the original model:** $Y(h_F) - Y(h)$.
- **Reconstruction damage:** $Y(\hat h) - Y(h)$, analyzed separately.
- **Feature effect in the replacement model:** $Y(h_{RF}) - Y(\hat h)$, with $h_{RF} = \hat h + \Delta d$ (a new arm).

$Y(h_F) - Y(\hat h)$ is not a feature effect: it also restores the reconstruction residual. With node normalization the edit is per node — $\Delta\, d$ times the dictionary scale times each node's own standard deviation — not one global vector; store the control's node statistics and test the mapping (T2 below). Scale and clamp deltas are computed from the control run, as shipped; say so in every record.

### 7.3 Operators and controls

- **Ablation:** clamp to zero only at nodes where the feature is active. Mean and resample ablations are different interventions; name the one used.
- **Doses:** quantiles of the feature's activation on positive discovery cases, applied only within a declared support (the AR object or a region). The shipped default — `add` at every valid node — is far from the training distribution and is not used.
- **Diagnostics for every patched state:** the realized delta, re-encoded feature changes, changes at the next captured site, distance from the training distribution.
- **Controls:** a norm-matched DiffMean direction (AR minus IVT-matched non-AR); a validation-tuned probe direction; covariance-aware random directions estimated on discovery activations (regularized); other learned features moved by the same activation quantile; a placebo region with the same support size; zero edits. Report which norm each control matches.
- **Perturbation cost and collateral damage**, recorded for every arm: cost is the Mahalanobis norm of the edit over its support under the discovery covariance at the site, matched across arms by construction; collateral damage is the RMS change outside the target region (over the declared fields) plus the change in non-target features at the next captured site.
- **Claim rule:** a larger endpoint change can mean more disruption, not a better explanation. An SAE-advantage claim therefore requires a larger on-target effect *at matched cost* than the DiffMean, probe and covariance-matched random directions, by a frozen superiority margin, with collateral damage no worse than theirs beyond a frozen non-inferiority margin (§8 test 5). Losing to them is a reportable result.

### 7.4 Calibrating the instrument

| Tier | Check | Pass |
|---|---|---|
| T0 | Toy: a frozen learned dictionary recovers planted patch effects, and zero edits are no-ops. Needs a small bridge, since `toy-dynamics` is `Intervenable` only | Exact, or within float tolerance |
| T1 | An output-site patch reproduces the expected denormalized, pre-corrector delta | Float tolerance |
| T2 | Hook identity at the real edit site: the realized activation equals $h+\Delta d$; editing `block_out(l)` and `block_in(l+1)` agree; ±ε responses are antisymmetric to O(ε²); the node-normalized mapping reproduces the intended reconstruction difference | Float tolerance |
| T3 | Known-response propagation: replace the edit site's activation at every node with a same-noise donor case's activation. Each block reads only its input and the shared noise, so every later captured block activation must equal the donor's. The output will not, because the decoder's big skip (`big_skip` is on by default and no config disables it) also reads the original input | Within the replay tolerance at every later captured site |
| Replay | Capture-only reruns on the same and different hardware; a zero-edit arm compared with the control | Tolerance curve recorded per endpoint, lead and device; bitwise equality claimed only where deterministic execution is enforced and shown to rerun bitwise |

T1–T3 establish that the instrument does what it says. They do not establish power. **Power** is a design calculation, recorded at the design freeze: from the calibration export's case-to-case and cluster-to-cluster variability, the number of clusters needed to detect the minimum useful effect (§8) at the Holm-adjusted level. A DiffMean direction's response is a scientific benchmark (§8 test 5), not a calibration: a direction chosen for detection can be orthogonal to what the model reads.

Sunlit land (SamudrACE feature 883, which tracks `SOLIN` and `LANDFRAC`) is a pre-registered *scientific* prediction, not a calibration: its correlation does not fix the sign of a decoder-vector edit through normalization, GELU and the corrector.

### 7.5 Failures and resources

- Invalid requests, non-finite archive inputs and malformed patches stay errors. A finite, valid patch that makes the model diverge is an outcome: a typed per-arm status (`ok`, `diverged_at_step`, `nonphysical`), persisted as each arm completes and reported by dose and control. Estimands: a failure rate, plus either a composite (failure mapped to a declared worst value) or a labeled while-stable estimand.
- Lead-1 envelope: one fp32 site is 99.5 MB, so with the default 23 rollouts per seed that is about 2.3 GB transferred per case and seed. Budget = cases × seeds × doses × arms × 99.5 MB per recorded site; record measured peak RSS and transfer in every study record. The 40-step figures (about 11.9 GB peak and 91.6 GB transferred per seed and site) belong to the propagation follow-up.

## 8. Pilot decision rule

The pilot is **exploratory**, whatever its size: its tests are decision criteria for whether to continue, not confirmatory claims (§6.3). The same tests, run on held-out clusters meeting the §6.3 requirements, become Milestone A's confirmatory tests.

**The pilot decision record** is frozen in two parts.

- **Part 1, the design freeze** (after calibration, before the bulk export): the §4 answers; populations, sampling and split rules; the candidate-selection procedure (at most three candidates, chosen on validation only); the margins (AUPRC margin over each null, non-inferiority margin to the dense probe, PCA margin, minimum useful effect δ_min, superiority margin Δ_on, collateral non-inferiority margin Δ_col); the held-out dead-fraction tolerance; the dose-quantile rule and support; the failure-rate threshold; the clustering design and N_min (§6.3); the power calculation (§7.4); budgets; the instrument time box. **Margins, δ_min, the tolerances and the time box are set a priori by the PI and the phenomenon specialist; data set only the replay tolerance, variability and precision** (from the calibration export).
- **Part 2, the candidate freeze** (after the discovery and validation fits, before any test access): basis hashes, candidate feature IDs, the dose values the rule produces, and the control covariance estimates.

**Tests**, each applied per candidate. Each yields a one-sided p-value from a wild cluster bootstrap (cluster = year, §6.3), obtained by test inversion; equivalently, a Holm-adjusted one-sided lower bound.

1. **Selection:** candidates come from validation only; the family for each test is the frozen candidates (m ≤ 3).
2. **Learned-vs-null test.** H0: the candidate's AUPRC within IVT deciles exceeds the best feature of the random-init SFNO SAE, and the best feature of the stacked-input SAE, by no more than the frozen margin. The null features are chosen on validation by the same procedure as the candidates. The candidate passes only if both comparisons reject (an intersection–union test: the larger of the two p-values is used, with no further adjustment).
3. **Probe-detection test.** H0: the candidate's AUPRC within IVT deciles falls below the dense probe's by more than the non-inferiority margin.
4. **Causal-effect test.** θ is the cluster mean of the case-mean signed regional lead-1 effect at the frozen dose, with the sign prespecified. H0: θ ≤ 0. The candidate passes if H0 is rejected *and* the estimate is at least δ_min.
5. **SAE-advantage test.** For each control direction c (DiffMean, probe, covariance-matched random), with all arms at matched cost (§7.3):
   - H0_on: on-target effect(candidate) − on-target effect(c) ≤ Δ_on;
   - H0_col: collateral(candidate) − collateral(c) ≥ Δ_col.

   The candidate passes only if every one of these nulls is rejected (intersection–union: the largest p-value is used).

**Holm across candidates**, separately for each test: order the candidates' p-values p₍₁₎ ≤ … ≤ p₍ₘ₎ and reject p₍ᵢ₎ while p₍ᵢ₎ ≤ α / (m − i + 1); stop at the first failure. The shipped random-direction rank p-value stays descriptive (§3.5). If a rank test is used instead, draw at least m/α − 1 directions (59 for m = 3, α = 0.05) and justify the exchangeable null.

A candidate whose within-decile AUPRC does not exceed the IVT-magnitude baseline is reported as an IVT-magnitude feature (§6.2).

**Routing**, applied in order; the first matching line decides.

- a. **Before the test look, instrument:** T1–T3 fail, or the power calculation shows the frozen cluster count cannot detect δ_min → fix the instrument or the design within the frozen time box. If either still fails when the box expires → **stop**, reported as an instrument or design failure; no scientific conclusion is drawn.
- b. **Before the test look, training quality (on validation):** held-out dead fraction above the frozen tolerance, or SAE worse than PCA at matched sparsity by more than the frozen margin → **redesign** (retrain, for example with AuxK), then redo the candidate freeze. One redesign is allowed; if the redesigned SAE fails (b) again → **stop**, reported as a training-quality result.
- c. **On the single test look**, judge each candidate separately against tests 2–5.
- d. At least one candidate passes tests 2–5 → **proceed** to Milestone A with the passing candidates only.
- e. At least one candidate passes tests 2 and 4, but none passes all of 2–5 → **narrow to a causal-effect study**: claims (1) and the causal-effect part of (3), with no SAE-advantage claim. Report which of tests 3 and 5 failed.
- f. At least one candidate passes test 2, but none passes test 4 → **narrow to a representation study**: claim (1), plus concept association with the label, explicitly named as association.
- g. Every candidate fails test 2 → **stop** this question and report it.

For d, e and f, the frozen report names the scope, and the group chooses at the Decision gate between a Milestone A of that scope and closing with the report. **Claim (2), encoding, is never inferred from detection scores or nuisance controls:** it requires admissible input interventions (§9).

The nuisance share and the response to prescribed SST forcing are diagnostics only: a genuine AR feature is seasonal and geographic.

## 9. Milestone A

Beyond the pilot, Milestone A requires:

- Confirmatory replication of the §8 tests that the pilot's route allows, on held-out cases meeting §6.3 (at least N_min year-clusters, a frozen clustering design, demonstrated coverage), with two detectors of different type.
- Claim (2), encoding, only if added explicitly: admissible input interventions (scaling IWV within the AR object while holding winds, or the reverse, kept physically consistent, or naturally occurring matched cases) showing the feature responds to the named structure beyond its nuisance correlates.
- A **minimal atlas** for the frozen candidates: event-deduplicated positives across activation quantiles; hard negatives (moist but weak transport, windy but dry, unrelated rainfall); split identities; baseline metrics; recall stratified by latitude, month, basin and intensity (an absorption audit); a state-vs-noise variance split for each feature; linked lead-1 results; for each feature a proposed name (a hypothesis), alternative explanations and its evidence stage (§1 claims 1–5); concept probes run separately on the reconstruction and on the SAE error term; validation by at least one field or detector other than the one that defined the label. The atlas exports without Streamlit.
- Lead-1 causal effects with the full control set, dose response, failures and clustered uncertainty.
- The basis evaluated on rollout latents at the leads used, as well as on teacher-forced latents.
- Fit metadata that records device, torch/CUDA/numpy versions and the deterministic-algorithms flag. The unit of reproducibility is the hashed basis file, not a refit.

**Gate:** every requirement listed above is met; T0–T3 and the replay check pass; frozen lead-1 effects, controls, support, clustered uncertainty and failures are reported; the pre-publication checklist (§11) is satisfied.

Activation-aware seed matching, group and subspace stability, split/merge across widths, sensitivity sweeps and response-type clustering are "Milestone A+", not requirements.

## 10. Package rules this plan changes, and when

| Rule (where) | Change | PR |
|---|---|---|
| Method order (`docs/package/nn.md`, `docs/package/index.md`): B-spline, then AuxK, then steering, then a cross-layer transcoder | Pilot first: safety, capture, a TopK pilot. AuxK is conditional; an MLP-sublayer transcoder comes before a cross-layer one; B-spline becomes a compared research variant | Note in PR1; lists rewritten when the pilot ends |
| Finiteness rule for steering fields (`latents/AGENTS.md`) | Divergence under a valid patch becomes a typed outcome; invalid inputs and patches are still refused | PR5 |
| Steering non-goals (`docs/package/steering.md`): one feature, no search, no gradients, no real model in xaig | The study runner composes `run_steering` for doses and controls; gradients stay a non-goal. The real model runs outside xaig's base and full tiers: in the external plugin (PR3a's default), or behind lazy imports if PR3a chooses the `ace` extra, in which case the non-goal is rewritten in PR3a | PR3a (home), PR5 (runner) |
| File formats (`latents/AGENTS.md`): only the basis `.npz` and the experiment record | New metadata become record kinds in `record.py`, validated and versioned; arrays and archives stay in adapters | PR4 (pilot decision, instrument card); PR5 (study spec, validator) |

No other rule changes are implied.

## 11. Instrument card and pre-publication checklist

**Instrument card.** A versioned record kind, one per checkpoint × site × patch operator × normalization × execution environment; every study record references its hash. It holds: identities and hashes (checkpoint file and which one, timestep, embedded config, normalization source, data manifest, fme commit); T1–T3 results, and a reference to the T0 result for the xaig version used (T0 does not depend on the checkpoint); the replay floor by endpoint, lead and device; measured noise sensitivity and the noise-affine magnitudes at the site; null-model results; interval coverage under synthetic dependence; fit and seed variability; corrector activity (precipitation ratio, dry-air offset, clamp counts); the working dose range; measured resources; known limitations. Recalibrate when any identity changes.

**Before any number is published**, a record validator checks: steering units (no cross-space dictionaries, per-node edits for node-normalized bases, realized changes recorded); artifact, site and sample identities; the emulator-training membership of every case; frozen selection and spec hash; endpoint, support and units; uncertainty with its number of clusters and, for confirmatory numbers, the demonstrated coverage at that count; calibration and replay floor; failure policy and counts; the exploratory or confirmatory label.

---

# Part II — reference

## A. Milestone B: a small path

**Entry gate:** MLP-sublayer replacement meets a frozen endpoint and site tolerance, and wrong-space writes are impossible under verified metadata.

**Transcoder target.** The MLP sublayer, `norm1_out → mlp_out`, as a skip transcoder with a matched dense-MLP baseline. A whole-block pointwise transcoder cannot see the spectral convolution and is not the default. Report what fraction of `mlp_out`, and of the block update, each component explains. Replace the sublayer in the original model and measure lead-1 downstream error, corrector effects included; evaluate captured inputs separately from replacement-induced ones.

**Spectral edges.** The filter is affine, so a perturbation propagates exactly through it at fixed layer-norm statistics (the bias cancels in differences). Whole paths are not exact: norm scales, GELU gates, noise affines and reconstruction errors need tested fixed-gate approximations.

**Paths.** Start from a lead-1 regional outcome and a few candidate site/feature groups. Interventions come first; the transcoder proposes contributions. Graph nodes carry checkpoint, site, feature or group, time and spatial support. Edges carry their evidence type — association, approximation or intervention — and the types are never interchangeable. Include spectral communication, skips, reconstruction errors, the corrector and the ocean prescription. Report suppressed and negative paths, redundancy and sensitivity to pruning. A `circuits` package appears only when reusable graph code exists, and its import edges then go into `tests/test_purity.py`.

**Mediation**, with natural direct and indirect effects (NDE, NIE) (A → B → Y, one case, all exogenous inputs replayed, B restored to that case's own value):
- pure NDE = Y(a, B(a₀)) − Y(a₀, B(a₀)) and total NIE = Y(a, B(a)) − Y(a, B(a₀)) sum to the total effect;
- total NDE and pure NIE = Y(a₀, B(a)) − Y(a₀, B(a₀)) form the other decomposition; do not mix the two;
- report the mediated interaction, total NIE − pure NIE, per case, and require each nested contrast to exceed the replay floor;
- when B is an SAE feature, restore it with the residual preserved and label the direct effect relative to that mediator;
- a different event's donor value is not B(a). For subspace or donor patches, check complement leakage, dormant components, alternative sites, and full-activation against subspace patches (Makelov et al.; the reply by Wu et al.);
- these are model counterfactuals, not identified atmospheric NDE and NIE.

**Interactions.** $I_{AB}=Y_{AB}-Y_A-Y_B+Y_0$ on a prespecified endpoint scale and a small factorial dose grid, normalized by $|Y_A-Y_0|+|Y_B-Y_0|$. Flag cases where clamps or the precipitation ratio bind.

**Gate:** §2.

## B. Causal framework

- The frozen emulator as an unrolled structural computation: $H_{\ell+1,t}=F_\ell(H_{\ell,t},U_t)$, $(X_{t+1},S_{t+1})=G(X_t,S_t,\mathcal F_t,U_t)$, $Y_q=Q(X_{0:T})$, with $S$ the corrector and RNG state and $U_t$ the shared noise. Forcing and noise are exogenous.
- Estimand: $\tau_q(a)=\mathbb E_{i,u}[Y_q(P_{j,a},i,u)-Y_q(P_0,i,u)]$ over a declared case population, paired noise and a prespecified endpoint; $P_0$ is the no-op for $h_F$ and the reconstruction for $h_{RF}$ (§7.2). Noise draws are averaged within a case; more seeds do not add events.
- A decoder-vector edit changes several features after re-encoding. Record intended and realized changes, and call the edit an activation patch unless independent control of the variable is shown.
- Associations, lagged prediction and attribution only propose hypotheses; seasonality, spatial dependence and shared inputs create them.
- Within-step effects (through the network) and across-step effects (through the state) are different claims. At lead 1 only the first exists.
- Compensation by later blocks, per-node norms, the moisture corrector or clamps can hide necessity. Where it matters, compare total effects with downstream-frozen direct effects.
- Necessity and sufficiency hold only for the tested patch family and population; redundant paths can make a group necessary when no single feature is.
- Invariance across regimes or checkpoints supports a mechanism but does not identify it.
- An out-of-support edit that breaks the model shows vulnerability, not the concept. Pointwise co-activation does not show spatial causation.
- Causal record fields: estimand and population; spec hash; identities; operator and doses; intended and realized changes; case and donor identities; noise pairing; endpoints, units and leads; artifact hashes; support and failures; resampling specification; interpretation limits.
- **Causal success at Milestone A** (restates §9; the tests are §8 tests 4 and 5): a frozen feature and prespecified patch produce a reproducible, dose-dependent lead-1 response of at least δ_min across held-out year-clusters; an SAE-advantage claim additionally beats the §7.3 controls at matched cost and collateral damage. **Circuit success (Milestone B, §2):** a small path prospectively predicts held-out responses to several interventions, with residual error and contradictions retained.

## C. Backlog and entry criteria

| Item | Enters when |
|---|---|
| Physical twins: the same perturbation in EAMv3 and the emulator; SST-patch Green's functions (cf. arXiv:2505.08742) | A cost probe is done (restart inventory of the AMIP members, spin-up per event, node-hours for one 5-day branch run) and a perturbation and ensemble design is approved. Weather-event twins and climate Green's functions answer different questions |
| Multi-step propagation | The lead-1 study is complete; lead-dependent replay floors, per-arm `RandomState` restore and `StepperState` replay (§3.3) are in place. The study measures feature-vs-random specificity by lead rather than assuming it converges or persists |
| Cross-checkpoint universality | Two checkpoints share a site definition and an evaluation population. If chosen as the first contribution (§1), it moves into Milestone A |
| Crosscoders | Paired multi-space samples, and a task where independent SAEs plus matching fall short. Report each stream's contribution; infer no causal graph from structure alone. A low decoder norm does not show absence; label unconstrained crosscoders retrospective |
| Cross-layer transcoders and larger graphs | MLP-sublayer replacement and small-path fidelity are established. Reads precede writes structurally; a smaller graph is not evidence of a more faithful one |
| Causal abstraction and interchange interventions | Candidate variables and their alignment are frozen, and small path tests justify fitting it. Prefer frozen SAE groups to learned rotations |
| Gradients (a VJP of a scalar endpoint) | A separate optional protocol, only when a path study needs it |
| B-spline (Cheon 2026) | Sparsity, capacity, held-out fidelity, stability and behavior matched against TopK. Spline gating keeps a linear decoder and settles neither identifiability nor a nonlinear mechanism |
| Matryoshka, spatial or multiscale, temporally predictive coders | Matryoshka: hierarchy against capacity effects. Spatial or multiscale: resolution and event transfer against a pointwise baseline. Temporally predictive: unseen trajectories, several lags, seasonality and identity baselines |
| Behavioral reconstruction metric (JVP/VJP) | Local predictions checked against finite interventions |
| Regime transfer | Support shift, event prevalence, reconstruction and behavior on new conditions |
| Multimodal analysis | The model actually ingests distinct modalities |
| SamudrACE | Ocean and coupling state are handled, with separate site and sample definitions |

## D. Persistence and package placement

Extend the `xaig.experiment-record` schema (version 1) and its helpers: add keys within a version, bump it for renames. The compact record links full artifacts by checksum. New kinds go through `record.py` (§10). Never guess missing identity; legacy bases stay readable through the explicit unverified path.

| Package | Responsibility |
|---|---|
| `core` | Registry, errors, hints for missing extras |
| `latents` | Sample and site identity, diagnostics, evaluation, matching, atlas data, records |
| `nn` | SAE and transcoder modules and their fitting |
| `adapters` (or the external plugin) | Archives, ACE capture, replay, patches |
| `figures`, `app` | Static panels; inspection of persisted results only |

## E. The v0.5.0 baseline

| Area | Shipped | Extended by |
|---|---|---|
| Correctness | Shared three-run masks, centered moments, identity checks, metrics of the exported dictionary, strict finite reads | PR1 |
| Provenance | Resolved hub revisions, basis content hashes (of content, not file bytes), reproduction commands | Instrument card, sample manifests |
| Evaluation | `evaluate_basis` (explained variance about the training mean; zero power gives NaN and JSON null), `fidelity_curve`, block/group/archive splits (gap counted in positions), redundancy | PR1b, PR4 |
| Seeds | `seed_stability`: one-to-one decoder matching; its random reference is a diagnostic | Milestone A+ |
| Steering | `Intervenable`, `Hook`, `Rollout`, `run_steering`; residual-preserving clamp and scale; `Steer.nodes` and `times` (the CLI exposes times, but not nodes or a start state); global-mean outcomes | PR5, the ACE adapter |
| Records | Evaluation and steering records, version 1; app views | §10–11 |
| Adapters | `latent-archive`, `bundle-dir`, `toy-dynamics`; contract tests | PR2–PR3b |

**Compatibility note.** v0.5.0 renamed British spellings without aliases. External scripts break only if they still use the old names: `--centred`, `centred=`, `analyse_region`, `normalise_decoder`, `centred` result keys, `INVALID_COLOUR` and `DARK_INVALID_COLOUR`. Repository code is already migrated; basis files, their hashes and version-1 records need no migration. Saved region or series JSON may need a key rename.

## F. References

These sources motivate methods; none guarantees a method transfers to ACE.

*Weather and climate interpretability*
1. MacMillan & Ouellette (2025), Towards mechanistic understanding in a data-driven weather model: internal activations reveal interpretable physical features, [arXiv:2512.24440](https://arxiv.org/abs/2512.24440) ([code](https://github.com/theodoremacmillan/graphcast-interpretability)): TopK SAEs with an AuxK-style loss on GraphCast, sparse logistic probes for AR and TC features, steering of a TC feature.
2. Cheon (2026), Beyond Linear Superposition: Discovering Climate Features in AI Weather Models with KAN-SAE, [arXiv:2605.17493](https://arxiv.org/abs/2605.17493): spline gating; single-author and unreplicated.
3. Tempest, Beylich & Craig (2026), Mechanistic Interpretability Tool for AI Weather Models, [arXiv:2604.20467](https://arxiv.org/abs/2604.20467) ([doi:10.1007/978-3-032-29915-4_10](https://doi.org/10.1007/978-3-032-29915-4_10); [code](https://github.com/ktempestuous/latent_space_visualiser_weather_models)): the region, channel, cosine-similarity and PCA workflow that `latents` reimplements.
4. Wu et al. (2025), Applying the ACE2 Emulator to SST Green's Functions for the E3SMv3 Global Atmosphere Model, [arXiv:2505.08742](https://arxiv.org/abs/2505.08742): ACE2 trained on EAMv3, compared with EAMv3 itself on GFMIP SST-patch runs; precedent for emulator-vs-E3SM forcing twins.

*SAEs, baselines and failure modes*

5. Gao et al. (2024), Scaling and evaluating sparse autoencoders, [arXiv:2406.04093](https://arxiv.org/abs/2406.04093): TopK, the AuxK dead-latent loss, evaluation.
6. Kantamneni et al. (2025), Are Sparse Autoencoders Useful? A Case Study in Sparse Probing, [arXiv:2502.16681](https://arxiv.org/abs/2502.16681): SAE probes against dense probes.
7. Wu et al. (2025), AxBench, [arXiv:2501.17148](https://arxiv.org/abs/2501.17148): DiffMean and probes against SAEs for detection and steering.
8. Heap et al. (2025), Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers, [arXiv:2501.17727](https://arxiv.org/abs/2501.17727): random-init controls.
9. Chanin et al. (2024), A is for Absorption, [arXiv:2409.14507](https://arxiv.org/abs/2409.14507): feature absorption.
10. Engels et al. (2024), Decomposing the Dark Matter of Sparse Autoencoders, [arXiv:2410.14670](https://arxiv.org/abs/2410.14670): structure in the SAE error.

*Patching, mediation and causal abstraction*

11. Vig et al. (2020), Causal mediation analysis for interpreting neural NLP, [arXiv:2004.12265](https://arxiv.org/abs/2004.12265).
12. Zhang & Nanda (2024), Towards Best Practices of Activation Patching, [arXiv:2309.16042](https://arxiv.org/abs/2309.16042); Heimersheim & Nanda (2024), How to use and interpret activation patching, [arXiv:2404.15255](https://arxiv.org/abs/2404.15255).
13. Goldowsky-Dill et al. (2023), Localizing model behavior with path patching, [arXiv:2304.05969](https://arxiv.org/abs/2304.05969).
14. Makelov et al. (2024), Is This the Subspace You Are Looking For?, [arXiv:2311.17030](https://arxiv.org/abs/2311.17030); reply by Wu et al., [arXiv:2401.12631](https://arxiv.org/abs/2401.12631).
15. Geiger et al. (2021), Causal Abstractions of Neural Networks, [arXiv:2106.02997](https://arxiv.org/abs/2106.02997); Geiger et al. (2022), Inducing Causal Structure for Interpretable Neural Networks (interchange intervention training), [ICML 2022](https://proceedings.mlr.press/v162/geiger22a.html).
16. Pearl (2001), Direct and indirect effects (UAI); the ICH E9(R1) addendum on estimands (2019), for intercurrent events.

*Statistics*

21. Cameron, Gelbach & Miller (2008), Bootstrap-Based Improvements for Inference with Clustered Errors, Review of Economics and Statistics 90(3) ([NBER Technical Working Paper 344](https://www.nber.org/papers/t0344), 2006): few-cluster inference and the wild cluster bootstrap.

*Transcoders and circuits (Milestone B and backlog)*

17. Dunefsky, Chlenski & Nanda (2024), Transcoders Find Interpretable LLM Feature Circuits, [arXiv:2406.11944](https://arxiv.org/abs/2406.11944).
18. Lindsey et al. (2024), [Sparse Crosscoders for Cross-Layer Features and Model Diffing](https://transformer-circuits.pub/2024/crosscoders/index.html).
19. Ameisen et al. (2025), [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html).
20. Schölkopf et al. (2021), Toward Causal Representation Learning, [arXiv:2102.11107](https://arxiv.org/abs/2102.11107).

*Backlog only (multimodal and hierarchical):* Goh et al. (2021), [Multimodal Neurons](https://distill.pub/2021/multimodal-neurons/); Templeton et al. (2024), [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html); Zaigrajew et al. (2025), [arXiv:2502.20578](https://arxiv.org/abs/2502.20578); Pach et al. (2025), [arXiv:2504.02821](https://arxiv.org/abs/2504.02821); Bussmann, Nabeshima, Karvonen & Nanda (2025), Learning Multi-Level Features with Matryoshka Sparse Autoencoders, [arXiv:2503.17547](https://arxiv.org/abs/2503.17547).

## G. Verification and limits

- xaig: the v0.5.0 suite, lint and format passed locally on Python 3.14.8 with torch. CI covers 3.11 and 3.13, with torch on 3.13 only. The defects in §3.5 were reproduced with small synthetic scripts against v0.5.0 (kept with the review record); PR1 turns each into a regression test.
- ACE fork: §3.2–3.4 come from reading `e273d543` and small numerical checks (layer-norm agreement, noise draw order), not from running a checkpoint.
- Not done: the documentation build (`mkdocs build --strict`), app layout, any checkpoint run, GPU replay, export at scale, data-custody transfer, E3SM twin, or Python 3.11 ACE integration.
- Citations: all arXiv IDs in §F were resolved against the arXiv API and the other URLs fetched, during the review of this revision.
- Review history: the first review's 89 surviving findings are not 89 pieces of independent evidence. IDs repeat across two of its lenses, and its completeness-critic items were never adversarially verified. Corrections accepted from the reconciliation and from the second review are folded into §3.
