"""Steering, against a system whose answer is planted.

``ToyDynamics`` is linear, and its latents do not read its state back, so what an edit
does to a field is worked out here by hand and not read off the code: adding ``a`` to
channel 0 of layer 1 moves temperature by ``gain * a`` on that step and by ``decay`` times
as much on each step after, and nothing else. Noise is additive and a function of the
seed, so a paired difference holds the edit and nothing else.
"""

from __future__ import annotations

import json
import shlex

import pytest

np = pytest.importorskip("numpy")

from xaig.adapters.toy_dynamics import FIELDS, ToyDynamics  # noqa: E402
from xaig.core import registry  # noqa: E402
from xaig.core.errors import RequestError  # noqa: E402
from xaig.latents import (  # noqa: E402
    Dictionary,
    Hook,
    Intervenable,
    Intervention,
    LatentSource,
    basis_hash,
    feature_direction,
    open_intervenable,
    run_steering,
    save_basis,
    save_result,
)
from xaig.latents.evaluate import jsonable  # noqa: E402

GAIN, DECAY, STEPS, T0, AMOUNT = 2.0, 0.5, 5, 1, 1.5


@pytest.fixture
def system():
    return ToyDynamics(masked=3, gain=GAIN, decay=DECAY)


def _weights(system):
    grid = system.grid()
    return grid.weights()[grid.valid], grid.valid


def _planted_series(first, amount=AMOUNT, steps=STEPS):
    """The temperature response of adding ``amount`` to the planted feature at ``first``."""
    return np.array([0.0] * first + [amount * GAIN * DECAY**k for k in range(steps - first)])


def _feature(layer=1, **kwargs):
    return Intervention(layer=layer, feature=0, **{"amount": AMOUNT, "times": (T0,), **kwargs})


def _control_latents(system, layer, seed, steps=STEPS):
    places = [(layer, t) for t in range(steps)]
    rollout = system.run(system.initial_state(), steps, (), noise_seed=seed, record=places)
    return {t: rollout.latents[(layer, t)] for t in range(steps)}


# -- the system ---------------------------------------------------------------


def test_the_toy_system_can_be_run_and_is_nothing_else(system):
    assert isinstance(system, Intervenable)
    assert not isinstance(system, LatentSource)


def test_noise_is_a_function_of_the_seed_alone(system):
    """Same seed, same run, to the bit; another seed, another run. This is what pairing
    leans on."""
    state = system.initial_state()
    a = system.run(state, STEPS, (), noise_seed=3)
    b = system.run(state, STEPS, (), noise_seed=3)
    c = system.run(state, STEPS, (), noise_seed=4)
    for name in FIELDS:
        valid = system.grid().valid
        assert np.array_equal(a.fields[name][:, valid], b.fields[name][:, valid])
        assert np.abs(a.fields[name][:, valid] - c.fields[name][:, valid]).max() > 1e-3


def test_fields_are_nan_where_the_mask_says_and_nowhere_else(system):
    out = system.run(system.initial_state(), 2, ())
    assert np.isnan(out.fields["temperature"][:, :3]).all()
    assert np.isfinite(out.fields["temperature"][:, 3:]).all()


def test_a_hook_edits_what_the_pass_continues_with(system):
    state = system.initial_state()
    base = system.run(state, 3, (), record=[(1, 1)])
    bump = Hook(1, 1, lambda h: h + np.float32(1.0) * np.eye(4, dtype=np.float32)[0])
    moved = system.run(state, 3, [bump], record=[(1, 1)])
    valid = system.grid().valid
    assert np.allclose(moved.latents[(1, 1)] - base.latents[(1, 1)], np.eye(4)[0], atol=1e-6)
    temperature = moved.fields["temperature"] - base.fields["temperature"]
    assert np.allclose(temperature[:, valid], [[0.0], [GAIN], [GAIN * DECAY]], atol=1e-5)


@pytest.mark.parametrize("place", [(2, 0), (0, 3), (1, -1)])
def test_a_place_that_does_not_exist_is_refused(system, place):
    with pytest.raises(RequestError, match="no place"):
        system.run(system.initial_state(), 3, [Hook(*place, lambda h: h)])


# -- the arms: planted answers ------------------------------------------------


def test_the_feature_arm_matches_the_planted_response_under_every_noise(system):
    """Temperature moves by gain * amount, then decays, whatever the noise: the paired
    difference holds the edit and nothing else. Nothing happens before the edit."""
    result = run_steering(
        system, system.planted_dictionary(), _feature(), steps=STEPS, seeds=(0, 1, 2), n_random=4
    )
    for seed in (0, 1, 2):
        paired = result.pairing("feature", seed).differences
        assert paired["temperature"][:T0].tolist() == [0.0] * T0  # exactly: before the edit
        assert np.allclose(paired["temperature"], _planted_series(T0), atol=1e-5)
        assert np.allclose(paired["moisture"], 0.0, atol=1e-5)
        assert np.allclose(paired["pressure"], 0.0, atol=1e-5)
    effect = result.effects["temperature"]
    assert effect.feature_response == pytest.approx(_planted_series(T0)[T0:].mean(), abs=1e-5)
    assert effect.feature_se == pytest.approx(0.0, abs=1e-5)  # one answer under every noise


def test_a_random_direction_does_not_do_what_the_feature_does(system):
    """A random unit direction r of the feature's length, added by the same amount, moves
    temperature by gain * amount * r[0] on the first step: strictly less, and different
    for each draw. The feature outranks them all."""
    result = run_steering(
        system, system.planted_dictionary(), _feature(), steps=STEPS, n_random=30, random_seed=7
    )
    first = []
    for draw in range(30):
        r = np.random.default_rng([7, draw]).normal(size=4)
        r /= np.linalg.norm(r)
        step = result.pairing("random", 0, draw).differences["temperature"][T0]
        assert step == pytest.approx(AMOUNT * GAIN * r[0], abs=1e-5)
        assert abs(step) < AMOUNT * GAIN
        first.append(step)
    assert len(set(np.round(first, 6))) == 30
    effect = result.effects["temperature"]
    assert (effect.rank, effect.percentile) == (1, 100.0)
    assert effect.p_value == pytest.approx(1 / 31)
    assert effect.effect_size > 2.0
    assert np.abs(effect.random_responses).max() < abs(effect.feature_response)


def test_a_feature_with_no_reach_into_a_field_does_not_stand_out_in_it(system):
    """The planted feature never touches pressure; random directions mostly do. Its
    response there is zero, and it is nearer the bottom of the draws than the top."""
    result = run_steering(system, system.planted_dictionary(), _feature(), steps=STEPS, n_random=30)
    effect = result.effects["pressure"]
    assert abs(effect.feature_response) < 1e-5
    assert effect.percentile < 10.0 and effect.rank > 27


def test_the_same_feature_one_layer_up_reaches_moisture_by_way_of_the_mixer(system):
    """Layer 0's channel 0 is swapped into channel 1 on the way to layer 1, which feeds
    moisture with weight one."""
    result = run_steering(
        system,
        system.planted_dictionary(layer=0),
        _feature(layer=0),
        steps=STEPS,
        n_random=3,
        record_layers=(0, 1),
    )
    paired = result.pairing("feature", 0).differences
    expected = np.array([0.0] * T0 + [AMOUNT * DECAY**k for k in range(STEPS - T0)])
    assert np.allclose(paired["moisture"], expected, atol=1e-5)
    assert np.allclose(paired["temperature"], 0.0, atol=1e-5)
    # the edit is seen at layer 0, then carried to layer 1 on the same step, and no later;
    # the RMS is over channels too, and one of four holds all of it
    rms = result.arm("feature", 0).latent_rms
    assert rms[(0, T0)] == pytest.approx(AMOUNT / 2, abs=1e-5)
    assert rms[(1, T0)] == pytest.approx(AMOUNT / 2, abs=1e-5)
    assert rms[(0, 0)] == 0.0 and rms[(1, T0 + 1)] == 0.0


def test_the_reconstruction_arm_differs_from_control_by_exactly_the_reconstruction_error(system):
    """The dictionary reads channels 0..2 through a ReLU, so it returns relu(h[:, :3]) and
    zero for channel 3. The error that leaves is worked out here, and is all the arm does."""
    basis = system.planted_dictionary()
    result = run_steering(system, basis, _feature(), steps=STEPS, seeds=(2,), n_random=2)
    h = _control_latents(system, 1, seed=2)[T0].astype(np.float64)
    recon = np.zeros_like(h)
    recon[:, :3] = np.maximum(h[:, :3], 0.0)
    error = recon - h
    assert np.abs(error).max() > 0.1  # there is something to find
    weights, valid = _weights(system)
    first_step = (error @ system.decoder)[valid]  # what x gains, node by node
    arm = result.pairing("reconstruction", 2)
    for i, name in enumerate(FIELDS):
        expected = weights @ first_step[:, i] * DECAY ** np.arange(STEPS - T0)
        assert np.allclose(arm.differences[name][T0:], expected, atol=1e-4)
        assert arm.differences[name][:T0].tolist() == [0.0] * T0
    rms = np.sqrt((error[valid] ** 2).mean(axis=1) @ weights)
    assert result.arm("reconstruction", 2).latent_rms[(1, T0)] == pytest.approx(rms, rel=1e-4)
    # and it is not the feature arm's doing: the feature arm leaves the error where it was
    assert result.pairing("feature", 2).differences["moisture"] == pytest.approx(0.0, abs=1e-5)


def test_paired_noise_cancels_to_exactly_zero_when_nothing_is_changed(system):
    """An amount of zero edits nothing. Every arm that is only an edit then equals its
    control to the bit, under noise: any difference left would be unpaired noise."""
    result = run_steering(
        system, system.planted_dictionary(), _feature(amount=0.0), steps=STEPS,
        seeds=(0, 1), n_random=3,
    )  # fmt: skip
    for pairing in result.pairings:
        if pairing.arm == "reconstruction":
            continue
        for name in FIELDS:
            assert pairing.differences[name].tolist() == [0.0] * STEPS
            assert pairing.response[name] == 0.0
    assert all(
        v == 0.0 for r in result.runs if r.arm != "reconstruction" for v in r.latent_rms.values()
    )
    # while the control itself is not still: one seed's noise is not another's
    one, two = (result.arm("control", s).outcomes["temperature"] for s in (0, 1))
    assert np.abs(one - two).max() > 1e-3


# -- modes, nodes, seeds ------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "amount", "delta"),
    [
        ("add", 0.7, lambda a, c: np.full_like(a, 0.7)),
        ("scale", 3.0, lambda a, c: 2.0 * a),
        ("clamp", 0.25, lambda a, c: 0.25 - a),
    ],
)
def test_scale_and_clamp_change_the_activation_as_stated(system, mode, amount, delta):
    """The feature's activation is relu(h[:, 0]); each mode is a change in it, node by
    node, and the first step's temperature moves by gain times the area-weighted mean."""
    result = run_steering(
        system, system.planted_dictionary(), _feature(mode=mode, amount=amount),
        steps=STEPS, n_random=2,
    )  # fmt: skip
    h = _control_latents(system, 1, seed=0)[T0].astype(np.float64)
    change = delta(np.maximum(h[:, 0], 0.0), h)
    weights, valid = _weights(system)
    step = result.pairing("feature", 0).differences["temperature"][T0]
    assert step == pytest.approx(GAIN * (weights @ change[valid]), abs=1e-4)


def test_scale_depends_on_the_noise_so_it_has_an_uncertainty_and_add_does_not(system):
    scaled = run_steering(
        system, system.planted_dictionary(), _feature(mode="scale", amount=3.0),
        steps=STEPS, seeds=(0, 1, 2, 3), n_random=2,
    )  # fmt: skip
    assert scaled.effects["temperature"].feature_se > 1e-3
    alone = run_steering(
        system, system.planted_dictionary(), _feature(), steps=STEPS, seeds=(0,), n_random=2
    )
    assert alone.effects["temperature"].feature_se is None


def test_an_edit_restricted_to_nodes_moves_only_their_share(system):
    weights, valid = _weights(system)
    nodes = (5, 7)
    result = run_steering(
        system, system.planted_dictionary(), _feature(nodes=nodes), steps=STEPS, n_random=2
    )
    share = weights[[n - 3 for n in nodes]].sum()  # nodes 0..2 are masked out
    step = result.pairing("feature", 0).differences["temperature"][T0]
    assert step == pytest.approx(AMOUNT * GAIN * share, abs=1e-5)


def test_acting_at_several_times_is_the_sum_of_acting_at_each(system):
    both = run_steering(
        system, system.planted_dictionary(), _feature(times=(1, 2)), steps=STEPS, n_random=2
    )
    expected = _planted_series(1) + _planted_series(2)
    assert np.allclose(both.pairing("feature", 0).differences["temperature"], expected, atol=1e-5)


def test_a_run_is_reproducible_from_its_settings(system):
    kwargs = {"steps": STEPS, "seeds": (0, 1), "n_random": 5, "random_seed": 3}
    a = run_steering(system, system.planted_dictionary(), _feature(), **kwargs)
    b = run_steering(system, system.planted_dictionary(), _feature(), **kwargs)
    assert json.dumps(jsonable(a.to_dict())) == json.dumps(jsonable(b.to_dict()))


# -- provenance and the file --------------------------------------------------


def test_the_result_carries_where_it_ran_and_with_which_basis(system, tmp_path):
    basis = system.planted_dictionary()
    result = run_steering(system, basis, _feature(), steps=STEPS, n_random=2)
    provenance = result.provenance
    assert provenance["source"] == "toy-dynamics" and provenance["checkpoint"] == "planted-0"
    assert provenance["options"]["masked"] == 3
    assert provenance["basis"]["sha256"] == basis_hash(basis)
    assert provenance["xaig"]
    path = save_result(tmp_path / "out" / "steer.json", result)
    written = json.loads(path.read_text())
    assert written["spec"]["intervention"]["amount"] == AMOUNT
    assert written["provenance"]["basis"]["sha256"] == basis_hash(basis)
    assert written["effects"]["temperature"]["rank"] == 1
    assert {r["arm"] for r in written["runs"]} == {"control", "reconstruction", "feature", "random"}


# -- refusals -----------------------------------------------------------------


class _Fake:
    """Enough of a basis to steer with, and to misbehave."""

    kind, n_features, n_channels = "fake", 1, 4

    def __init__(self, system, *, reconstruct=None, transform=None):
        self.meta = {"fitted_on": {"provenance": system.info().identity(), "layer": 1}}
        self._reconstruct = reconstruct or (lambda h: h)
        self._transform = transform

    def directions(self):
        return np.eye(4)[:1]

    def reconstruct(self, h):
        return self._reconstruct(h)

    def transform(self, h, features=None):
        return self._transform(h)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"seeds": ()}, "seeds"),
        ({"seeds": (1, 1)}, "seeds"),
        ({"n_random": 0}, "random"),
        ({"steps": 0}, "steps"),
        ({"fields": ()}, "no fields"),
        ({"fields": ("salinity",)}, "no field 'salinity'"),
        ({"record_layers": (9,)}, "cannot record"),
    ],
)
def test_what_cannot_be_asked_is_refused_before_anything_runs(system, change, match):
    kwargs = {"steps": STEPS, "n_random": 2, **change}
    with pytest.raises(RequestError, match=match):
        run_steering(system, system.planted_dictionary(), _feature(), **kwargs)


@pytest.mark.parametrize(
    ("intervention", "match"),
    [
        (_feature(layer=5), "no layer 5"),
        (Intervention(layer=1, feature=9), "feature 9"),
        (_feature(times=(STEPS,)), "cannot act"),
        (_feature(times=(-1,)), "cannot act"),
        (_feature(nodes=(0, 1)), "masked out"),
        (_feature(nodes=(999,)), "nodes must lie"),
    ],
)
def test_an_intervention_that_does_not_exist_is_refused(system, intervention, match):
    with pytest.raises(RequestError, match=match):
        run_steering(system, system.planted_dictionary(), intervention, steps=STEPS, n_random=2)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mode": "multiply"}, "mode must be"),
        ({"amount": float("nan")}, "finite"),
        ({"amount": float("inf")}, "finite"),
        ({"times": ()}, "at least one time"),
        ({"nodes": ()}, "no nodes"),
    ],
)
def test_an_intervention_is_checked_when_it_is_made(kwargs, match):
    with pytest.raises(RequestError, match=match):
        Intervention(layer=1, feature=0, **kwargs)


def test_a_basis_fitted_on_another_network_does_not_steer_this_one(system):
    basis = system.planted_dictionary()
    other = Dictionary(
        encoder=basis.encoder, encoder_bias=basis.encoder_bias, decoder=basis.decoder,
        decoder_bias=basis.decoder_bias, input_mean=basis.input_mean,
        meta={"fitted_on": {"provenance": {**system.info().identity(), "checkpoint": "x"},
                            "layer": 1}},
    )  # fmt: skip
    with pytest.raises(RequestError, match="not fitted here"):
        run_steering(system, other, _feature(), steps=STEPS, n_random=2)
    with pytest.raises(RequestError, match="layer 1 against the layer 0"):
        run_steering(system, system.planted_dictionary(layer=0), _feature(), steps=STEPS)


def test_a_transcoder_cannot_steer_the_layer_it_reads():
    wide = Dictionary(
        encoder=np.eye(4, dtype=np.float32)[:2], encoder_bias=np.zeros(2, dtype=np.float32),
        decoder=np.ones((2, 5), dtype=np.float32), decoder_bias=np.zeros(5, dtype=np.float32),
        input_mean=np.zeros(4, dtype=np.float32),
    )  # fmt: skip
    with pytest.raises(RequestError, match="another layer"):
        feature_direction(wide, 0)


def test_a_direction_that_is_not_finite_is_refused():
    broken = Dictionary(
        encoder=np.eye(4, dtype=np.float32)[:1], encoder_bias=np.zeros(1, dtype=np.float32),
        decoder=np.array([[np.inf, 0, 0, 0]], dtype=np.float32),
        decoder_bias=np.zeros(4, dtype=np.float32), input_mean=np.zeros(4, dtype=np.float32),
    )  # fmt: skip
    with pytest.raises(RequestError, match="not finite"):
        feature_direction(broken, 0)


def test_an_edit_that_is_not_finite_or_changes_the_shape_is_refused_where_it_happens(system):
    nan = _Fake(system, transform=lambda h: np.full((h.shape[0], 1), np.nan))
    with pytest.raises(RequestError, match="not finite on valid nodes"):
        run_steering(system, nan, _feature(mode="scale", amount=2.0), steps=STEPS, n_random=1)
    narrow = _Fake(system, reconstruct=lambda h: h[:, :2])
    with pytest.raises(RequestError, match="returned"):
        run_steering(system, narrow, _feature(), steps=STEPS, n_random=1)


def test_a_field_that_is_nan_on_a_valid_node_is_refused_not_averaged(system):
    class Diverged(ToyDynamics):
        def run(self, *args, **kwargs):
            out = super().run(*args, **kwargs)
            out.fields["temperature"][2, 10] = np.nan  # node 10 is valid
            return out

    broken = Diverged(masked=3)
    with pytest.raises(RequestError, match="not finite on valid nodes at step 2"):
        run_steering(broken, broken.planted_dictionary(), _feature(), steps=STEPS, n_random=1)


def test_a_system_that_cannot_be_run_is_not_steered():
    class Reader:
        def info(self):
            raise AssertionError("must not get that far")

    with pytest.raises(RequestError, match="Intervenable"):
        run_steering(Reader(), None, _feature(), steps=1)


# -- through the registry and the command ------------------------------------


def test_the_toy_system_is_a_registered_adapter_that_reads_nothing():
    assert "toy-dynamics" in registry.available()
    built = open_intervenable(None, "toy-dynamics", masked=2)
    assert isinstance(built, Intervenable) and built.info().options["masked"] == 2
    with pytest.raises(Exception, match="does not take a source"):
        open_intervenable("somewhere", "toy-dynamics")
    with pytest.raises(Exception, match="does not accept option"):
        open_intervenable(None, "toy-dynamics", maskd=2)


def test_an_adapter_that_cannot_be_run_is_not_offered_as_one():
    class Reader:
        def __init__(self, source):
            pass

    registry.register("reader-only", Reader)
    try:
        with pytest.raises(Exception, match="does not implement Intervenable"):
            open_intervenable("x", "reader-only")
    finally:
        registry.unregister("reader-only")


def _steer_args(basis, *extra):
    return [
        "latents", "steer", "--adapter", "toy-dynamics", "--adapter-option", "masked=3",
        "--basis", str(basis), "--layer", "1", "--feature", "0", "--amount", "1.5", "--time", "1",
        "--steps", "5", "--seeds", "0,1", "--random-draws", "12", *extra,
    ]  # fmt: skip


def test_the_command_reports_the_planted_effect_and_a_command_that_reproduces_it(tmp_path):
    from click.testing import CliRunner

    from xaig._cli import cli

    basis = tmp_path / "planted.npz"
    save_basis(basis, ToyDynamics(masked=3).planted_dictionary())
    run = CliRunner()

    done = run.invoke(cli, _steer_args(basis, "--json", "--out", str(tmp_path / "r" / "s.json")))
    assert done.exit_code == 0, done.output
    payload = json.loads(done.output)
    assert payload == json.loads((tmp_path / "r" / "s.json").read_text())
    effect = payload["effects"]["temperature"]
    assert effect["feature_response"] == pytest.approx(_planted_series(T0)[T0:].mean(), abs=1e-4)
    assert (effect["rank"], effect["n_draws"]) == (1, 12)
    assert payload["provenance"]["basis"]["status"] == "verified"
    assert payload["provenance"]["options"] == {**ToyDynamics(masked=3).info().options}

    text = run.invoke(cli, _steer_args(basis))
    assert text.exit_code == 0, text.output
    assert "temperature" in text.output and "1/13" in text.output
    sha = payload["provenance"]["basis"]["sha256"]
    line = next(x for x in text.output.splitlines() if x.startswith("reproduce:"))
    assert (
        f"--basis-sha256 {sha}" in line and "masked=3" in line and "--adapter toy-dynamics" in line
    )
    # and the printed line, run again, says the same thing
    again = run.invoke(cli, [*shlex.split(line.removeprefix("reproduce: xaig ")), "--json"])
    assert again.exit_code == 0, again.output
    assert json.loads(again.output)["effects"] == payload["effects"]


def test_the_command_refuses_a_basis_whose_content_is_not_the_one_pinned(tmp_path):
    from click.testing import CliRunner

    from xaig._cli import cli

    basis = tmp_path / "planted.npz"
    save_basis(basis, ToyDynamics(masked=3).planted_dictionary())
    out = CliRunner().invoke(cli, _steer_args(basis, "--basis-sha256", "0" * 64))
    assert out.exit_code != 0 and "sha256" in out.output
    bad = CliRunner().invoke(cli, [*_steer_args(basis)[:-2], "--random-draws", "0"])
    assert bad.exit_code != 0 and "random" in bad.output
