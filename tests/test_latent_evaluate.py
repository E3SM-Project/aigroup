"""Evaluating a frozen basis on held-out times, with answers planted in the data.

The fixture is a rank-four layer: every node is two of four unit directions of an
orthonormal frame, with known amounts. A dictionary holding that frame is perfect on
it, one that lacks a direction loses exactly that direction's energy, and the numbers
asserted below are worked out from the planted amounts, not read off the code.
"""

from __future__ import annotations

import itertools
import json

import pytest

np = pytest.importorskip("numpy")

from conftest import MemorySource, write_latent_archive  # noqa: E402
from xaig.core.errors import RequestError  # noqa: E402
from xaig.latents import (  # noqa: E402
    PCA,
    Dictionary,
    accumulate_moments,
    evaluate_basis,
    fidelity_curve,
    load_basis,
    open_source,
    pca_from_moments,
    save_basis,
    save_result,
    seed_stability,
    split_archives,
    split_groups,
    split_time_blocks,
)
from xaig.latents.evaluate import assign, match_features, redundancy  # noqa: E402

N_TIMES, N_NODES, WIDTH, N_REAL = 12, 48, 6, 4
TIMES = [f"t{i}" for i in range(N_TIMES)]
NOISE = 1e-4  # float32 leaves ~1e-8 where a feature is "zero": active means above this
OFFSET = 2.0  # carried along a direction the dictionary does not have, in the held-out times


def _frame():
    q, _ = np.linalg.qr(np.random.default_rng(5).normal(size=(WIDTH, WIDTH)))
    return q.T  # rows are orthonormal directions; 0..3 are planted, 4 and 5 are not


def _planted(offset_from=None):
    """``(n_times, n_nodes, WIDTH)``: two active features per node, amounts in 1..3."""
    rng = np.random.default_rng(11)
    frame = _frame()
    data = np.empty((N_TIMES, N_NODES, WIDTH))
    for t in range(N_TIMES):
        amounts = np.zeros((N_NODES, N_REAL))
        for node in range(N_NODES):
            amounts[node, rng.choice(N_REAL, size=2, replace=False)] = rng.uniform(1.0, 3.0, 2)
        data[t] = amounts @ frame[:N_REAL]
        if offset_from is not None and t >= offset_from:
            data[t] += OFFSET * frame[4]
    return data


def _perfect(meta=None):
    """The planted frame as a dictionary: features 0..3 are the planted directions and
    answer to nothing else; 4 and 5 are never able to fire."""
    frame = _frame()
    return Dictionary(
        encoder=frame,
        encoder_bias=np.array([0.0] * N_REAL + [-1e3] * (WIDTH - N_REAL)),
        decoder=frame,
        decoder_bias=np.zeros(WIDTH),
        input_mean=np.zeros(WIDTH),
        meta=meta or {},
    )


def _source(**kwargs):
    return MemorySource({0: _planted(**kwargs)})


def _split(**kwargs):
    return split_time_blocks(TIMES, n_blocks=4, test_blocks=(-1,), gap=1, **kwargs)


# -- splits ------------------------------------------------------------------------


def test_a_time_split_is_contiguous_buffered_and_explicit():
    split = split_time_blocks(TIMES, n_blocks=4, test_blocks=(-1,), gap=1)
    assert split.test == ("t9", "t10", "t11")
    assert split.buffer == ("t8",)
    assert split.train == tuple(TIMES[:8])
    assert split.to_dict()["blocks"] == [TIMES[0:3], TIMES[3:6], TIMES[6:9], TIMES[9:12]]

    middle = split_time_blocks(TIMES, n_blocks=4, test_blocks=(1,), gap=2)
    assert middle.test == ("t3", "t4", "t5")
    assert middle.buffer == ("t1", "t2", "t6", "t7")  # on both sides, two deep
    assert middle.train == ("t0", "t8", "t9", "t10", "t11")
    held = [TIMES.index(t) for t in middle.test]
    assert all(abs(TIMES.index(t) - h) > 2 for t in middle.train for h in held)
    every = middle.train + middle.buffer + middle.test
    assert sorted(every, key=TIMES.index) == TIMES  # nothing lost, nothing twice


def test_a_split_that_would_leak_or_leave_nothing_is_refused():
    with pytest.raises(RequestError, match="gap must be at least 1"):
        split_time_blocks(TIMES, gap=0)
    with pytest.raises(RequestError, match="nothing to fit on"):
        split_time_blocks(TIMES, n_blocks=4, test_blocks=(0, 1, 2), gap=3)
    with pytest.raises(RequestError, match="no block 4"):
        split_time_blocks(TIMES, n_blocks=4, test_blocks=(4,))
    with pytest.raises(RequestError, match="cannot be cut"):
        split_time_blocks(TIMES[:3], n_blocks=5)


def test_trajectories_are_held_out_whole():
    groups = {t: "a" if i % 2 else "b" for i, t in enumerate(TIMES)}
    split = split_groups(TIMES, groups, ["a"])
    assert set(split.test) == {t for t in TIMES if groups[t] == "a"} and not split.buffer
    assert set(split.train) == {t for t in TIMES if groups[t] == "b"}
    with pytest.raises(RequestError, match="no group 'c'"):
        split_groups(TIMES, groups, ["c"])
    with pytest.raises(RequestError, match="no group given"):
        split_groups(TIMES, {"t0": "a"}, ["a"])


# -- a frozen dictionary, on both sides ---------------------------------------------


def test_a_dictionary_that_holds_the_planted_frame_is_perfect_on_held_out_times():
    source, split = _source(), _split()
    result = evaluate_basis(
        _perfect(), source, split, layer=0, active_above=NOISE, allow_unverified_basis=True
    )
    for side in (result.train, result.test):
        assert side.explained_variance == pytest.approx(1.0, abs=1e-6)
        assert side.mse == pytest.approx(0.0, abs=1e-8)
        assert side.mean_active_features == pytest.approx(2.0, abs=1e-6)  # two a node, every node
        assert side.l0_fraction == pytest.approx(2.0 / WIDTH, abs=1e-6)
        assert side.dead == (4, 5) and side.dead_fraction == pytest.approx(2.0 / WIDTH)
    assert len(result.test.times) == 3 and result.test.times == split.test
    assert result.dormant == ()
    assert result.redundancy.max_cosine.max() < 1e-5  # an orthonormal frame
    assert result.redundancy.n_near_duplicates == 0 and result.redundancy.n_pairs == 0


def test_held_out_times_are_scored_by_what_the_frozen_dictionary_does_to_them():
    """From t9 on, every node carries +OFFSET along a direction feature 4 would have had
    and cannot fire for. Its energy is all error, and the other directions are untouched."""
    source, split = _source(offset_from=9), _split()
    result = evaluate_basis(
        _perfect(), source, split, layer=0, active_above=NOISE, allow_unverified_basis=True
    )
    assert result.train.explained_variance == pytest.approx(1.0, abs=1e-6)

    power = 0.0
    weights = source.grid().weights()
    for time in split.test:
        power += weights @ (source.load(time, 0).astype(np.float64) ** 2).sum(axis=1)
    power /= len(split.test)
    assert result.test.mse == pytest.approx(OFFSET**2 / WIDTH, rel=1e-5)
    assert result.test.explained_variance == pytest.approx(1.0 - OFFSET**2 / power, rel=1e-5)
    assert result.test.explained_variance < 0.95
    assert result.test.mean_active_features == pytest.approx(2.0, abs=1e-6)  # still two a node
    assert result.test.dead == (4, 5)  # the offset does not wake what cannot fire


def test_a_feature_that_only_the_training_times_use_is_dormant_held_out():
    rng = np.random.default_rng(2)
    data = rng.uniform(1.0, 2.0, (N_TIMES, N_NODES, 2))
    data[9:, :, 1] = 0.0  # a second channel that goes silent from t9 on
    identity = np.eye(2)
    dictionary = Dictionary(identity, np.zeros(2), identity, np.zeros(2), np.zeros(2))
    result = evaluate_basis(
        dictionary, MemorySource({0: data}), _split(), layer=0, allow_unverified_basis=True
    )
    assert result.train.dead == () and result.test.dead == (1,) and result.dormant == (1,)
    assert result.test.firing_rate[0] == pytest.approx(1.0)


def test_a_basis_that_saw_a_held_out_time_is_refused():
    source, split = _source(), _split()
    leaky = accumulate_moments(source, layer=0, times=list(split.train) + ["t9"])
    with pytest.raises(RequestError, match="fitted on 1 held-out time.*'t9'"):
        evaluate_basis(
            pca_from_moments(leaky, 2), source, split, layer=0, allow_unverified_basis=True
        )
    honest = pca_from_moments(accumulate_moments(source, layer=0, times=list(split.train)), 2)
    assert evaluate_basis(honest, source, split, layer=0).fitted_on == "train"
    elsewhere = pca_from_moments(accumulate_moments(source, layer=0, times=["t0", "t1"]), 2)
    assert evaluate_basis(elsewhere, source, split, layer=0).fitted_on == "other"
    said_nothing = evaluate_basis(_perfect(), source, split, layer=0, allow_unverified_basis=True)
    assert said_nothing.fitted_on == "unknown"


def test_an_unverified_basis_is_not_accepted_without_saying_so():
    with pytest.raises(RequestError, match="unverified"):
        evaluate_basis(_perfect(), _source(), _split(), layer=0)


def test_values_that_are_not_numbers_are_refused_not_averaged():
    data = _planted()
    data[10, 3, 0] = np.nan
    with pytest.raises(RequestError, match="not finite"):
        evaluate_basis(
            _perfect(), MemorySource({0: data}), _split(), layer=0, allow_unverified_basis=True
        )


def test_the_held_out_side_may_be_another_archive(tmp_path):
    first = open_source(write_latent_archive(tmp_path / "a"))
    second = open_source(write_latent_archive(tmp_path / "b"))
    split = split_archives(first, second, layer=2)
    assert split.scheme == "archives" and split.test_source == str(tmp_path / "b")
    basis = pca_from_moments(accumulate_moments(first, layer=2), 3)
    result = evaluate_basis(basis, first, split, layer=2, test_source=second)
    assert result.train.explained_variance == pytest.approx(
        result.test.explained_variance, abs=1e-6
    )
    assert result.provenance["test_source"]["source"] == str(tmp_path / "b")
    with pytest.raises(RequestError, match="goes with, and only with"):
        evaluate_basis(basis, first, split, layer=2)
    fitted_on_second = pca_from_moments(accumulate_moments(second, layer=2), 3)
    with pytest.raises(RequestError, match="the held-out archive"):
        evaluate_basis(fitted_on_second, first, split, layer=2, test_source=second)


# -- redundancy ----------------------------------------------------------------------


def test_duplicated_decoder_rows_are_flagged_and_an_opposite_is_not():
    frame = _frame()
    decoder = np.vstack([frame[:3], frame[0], -frame[1]])  # row 3 copies 0; row 4 negates 1
    decoder[3] += 1e-3 * frame[5]  # a copy, near enough
    dictionary = Dictionary(decoder, np.zeros(5), decoder, np.zeros(WIDTH), np.zeros(WIDTH))
    found = redundancy(dictionary, threshold=0.99)
    assert found.n_near_duplicates == 2 and found.n_pairs == 1
    assert found.nearest[0] == 3 and found.nearest[3] == 0
    assert found.max_cosine[0] > 0.9999
    assert found.max_cosine[1] < 0.01  # its opposite is not its duplicate
    assert found.n_zero == 0
    assert redundancy(dictionary, threshold=0.99).to_dict()["n_pairs"] == 1


def test_exact_copies_meet_a_threshold_of_one_at_any_width():
    rng = np.random.default_rng(3)
    decoder = rng.normal(size=(50, 300))
    decoder = np.vstack([decoder, decoder[:5]])  # five exact copies: ten rows in pairs
    dictionary = Dictionary(decoder, np.zeros(55), decoder, np.zeros(300), np.zeros(300))
    found = redundancy(dictionary, threshold=1.0)
    assert found.n_near_duplicates == 10 and found.n_pairs == 5
    partner, cosine = match_features(dictionary, dictionary)
    assert partner.tolist() == list(range(55)) and cosine.min() > 1 - 1e-12


# -- the matching -------------------------------------------------------------------


def test_the_assignment_refuses_more_rows_than_columns():
    with pytest.raises(RequestError, match="no more rows"):
        assign(np.zeros((3, 2)))


@pytest.mark.parametrize("shape", [(5, 5), (4, 7), (6, 6)])
def test_the_assignment_is_the_cheapest_there_is(shape):
    cost = np.random.default_rng(shape[0] * 10 + shape[1]).normal(size=shape)
    taken = assign(cost)
    assert len(set(taken.tolist())) == shape[0]
    best = min(
        sum(cost[row, column] for row, column in enumerate(columns))
        for columns in itertools.permutations(range(shape[1]), shape[0])
    )
    assert sum(cost[row, column] for row, column in enumerate(taken)) == pytest.approx(best)


def _random_dictionary(seed, n_features=8, n_channels=32):
    rng = np.random.default_rng(seed)
    decoder = rng.normal(size=(n_features, n_channels))
    decoder /= np.linalg.norm(decoder, axis=1, keepdims=True)
    return Dictionary(
        decoder.copy(), np.zeros(n_features), decoder, np.zeros(n_channels), np.zeros(n_channels)
    )


def test_identical_seeds_match_perfectly_and_a_shuffle_is_undone():
    one = _random_dictionary(0)
    order = np.random.default_rng(1).permutation(8)
    shuffled = Dictionary(
        one.encoder[order], one.encoder_bias, one.decoder[order],
        one.decoder_bias, one.input_mean,
    )  # fmt: skip
    partner, cosine = match_features(one, shuffled)
    assert cosine == pytest.approx(1.0) and (order[partner] == np.arange(8)).all()

    result = seed_stability([one, shuffled, one], allow_unverified=True)
    assert result.fraction_recurring == pytest.approx(1.0)
    assert result.recurrence == (1.0, 1.0, 1.0)
    assert result.matched.size == 3 * 2 * 8  # every ordered pair
    assert result.to_dict()["matched_similarity"]["min"] == pytest.approx(1.0)


def test_unrelated_seeds_do_not_recur_and_chance_is_reported():
    result = seed_stability(
        [_random_dictionary(1), _random_dictionary(2)], threshold=0.9, allow_unverified=True
    )
    assert result.fraction_recurring == 0.0 and result.recurrence == (0.0, 0.0)
    assert result.matched.max() < 0.9
    # the forced pairing of unrelated directions is what a match is to be measured against
    assert np.median(result.chance) == pytest.approx(np.median(result.matched), abs=0.15)
    assert 0.0 < np.median(result.chance) < 0.9


def test_stability_refuses_to_line_up_what_is_not_one_place_in_one_network():
    one, two = _random_dictionary(1), _random_dictionary(2)
    with pytest.raises(RequestError, match="not recorded"):
        seed_stability([one, two])
    with pytest.raises(RequestError, match="different shapes"):
        seed_stability([one, _random_dictionary(3, n_features=9)], allow_unverified=True)
    at = lambda layer, ckpt: {  # noqa: E731
        "fitted_on": {
            "layer": layer,
            "provenance": {"model": "m", "component": "c", "checkpoint": ckpt},
        }
    }
    with_meta = lambda base, meta: Dictionary(  # noqa: E731
        base.encoder, base.encoder_bias, base.decoder, base.decoder_bias, base.input_mean, meta=meta
    )
    with pytest.raises(RequestError, match="checkpoint 'a' against 'b'"):
        seed_stability([with_meta(one, at(2, "a")), with_meta(two, at(2, "b"))])
    with pytest.raises(RequestError, match="layers \\[2, 3\\]"):
        seed_stability([with_meta(one, at(2, "a")), with_meta(two, at(3, "a"))])
    same = seed_stability([with_meta(one, at(2, "a")), with_meta(two, at(2, "a"))])
    assert not same.same_training_times  # neither says what it was fitted on
    with pytest.raises(RequestError, match="at least two"):
        seed_stability([one])


# -- fidelity against sparsity ---------------------------------------------------------


def test_pca_needs_the_planted_rank_and_a_dictionary_needs_fewer_features_active():
    source, split = _source(), _split()
    curve = fidelity_curve(
        source, split, layer=0, pca_components=[1, 2, 3, 4, 5],
        dictionaries=[_perfect()], labels=["frame"], active_above=NOISE,
        allow_unverified_basis=True,
    )  # fmt: skip
    pca = curve.method("pca")
    assert [p.setting["k"] for p in pca] == [1, 2, 3, 4, 5]
    sparsity = [p.evaluation.test.mean_active_features for p in pca]
    # k active at every node; a fifth component of rank-four data scores nothing above noise
    assert sparsity == pytest.approx([1.0, 2.0, 3.0, 4.0, 4.0])
    fidelity = [p.evaluation.test.explained_variance for p in pca]
    assert fidelity == sorted(fidelity) and fidelity[3] == pytest.approx(1.0, abs=1e-6)
    assert fidelity[1] < 0.9  # two components of four planted directions cannot do it
    assert all(p.evaluation.fitted_on == "train" for p in pca)  # PCA and moments: train only

    (point,) = curve.method("sae")
    assert point.label == "frame"
    assert point.evaluation.test.mean_active_features == pytest.approx(2.0, abs=1e-6)
    assert point.evaluation.test.explained_variance == pytest.approx(1.0, abs=1e-6)
    assert point.pca_at_same_sparsity == pytest.approx(fidelity[1], abs=1e-9)
    assert point.pca_at_same_sparsity < point.evaluation.test.explained_variance


def test_a_dictionary_outside_the_pca_range_has_nothing_to_be_set_against():
    source, split = _source(), _split()
    curve = fidelity_curve(
        source, split, layer=0, pca_components=[3, 4],
        dictionaries=[_perfect()], allow_unverified_basis=True,
    )  # fmt: skip
    assert curve.method("sae")[0].pca_at_same_sparsity is None  # two active; the curve starts at 3


def test_the_curve_refuses_a_dictionary_trained_on_held_out_times_and_a_transcoder():
    source, split = _source(), _split()
    leaky = _perfect({"fitted_on": {"times": ["t0", "t10"]}})
    with pytest.raises(RequestError, match="held-out"):
        fidelity_curve(
            source, split, layer=0, pca_components=[2], dictionaries=[leaky],
            allow_unverified_basis=True,
        )  # fmt: skip
    frame = _frame()
    transcoder = Dictionary(
        frame, np.zeros(WIDTH), frame, np.zeros(WIDTH), np.zeros(WIDTH),
        output_mean=np.zeros(WIDTH), output_scale=1.0,
    )  # fmt: skip
    with pytest.raises(RequestError, match="transcoder"):
        fidelity_curve(source, split, layer=0, pca_components=[2], dictionaries=[transcoder])
    with pytest.raises(RequestError, match="pca_components is empty"):
        fidelity_curve(source, split, layer=0, pca_components=[])


def test_a_transcoder_is_measured_against_the_layer_it_writes():
    """Layer 1 is layer 0 doubled; a transcoder that doubles is exact."""
    data = _planted()
    source = MemorySource({0: data, 1: 2.0 * data})
    frame = _frame()
    transcoder = Dictionary(
        frame, np.array([0.0] * N_REAL + [-1e3] * 2), frame, np.zeros(WIDTH), np.zeros(WIDTH),
        output_mean=np.zeros(WIDTH), output_scale=2.0,
    )  # fmt: skip
    result = evaluate_basis(
        transcoder, source, _split(), layer=0, target_layer=1, allow_unverified_basis=True
    )
    assert result.test.explained_variance == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(RequestError, match="give target_layer"):
        evaluate_basis(transcoder, source, _split(), layer=0, allow_unverified_basis=True)


# -- serialisation and provenance --------------------------------------------------------


def test_results_are_json_with_the_split_and_the_basis_that_made_them(tmp_path):
    source, split = _source(), _split()
    path = save_basis(tmp_path / "frame.npz", _perfect())
    basis = load_basis(path)
    curve = fidelity_curve(
        source, split, layer=0, pca_components=[2, 4], dictionaries=[basis],
        allow_unverified_basis=True,
    )  # fmt: skip
    out = save_result(tmp_path / "out" / "curve.json", curve)
    loaded = json.loads(out.read_text())
    assert loaded["split"]["test"] == list(split.test) and loaded["split"]["buffer"] == ["t8"]
    assert loaded["provenance"]["xaig"] and loaded["provenance"]["model"] == "mem"
    sae = [p for p in loaded["points"] if p["method"] == "sae"][0]
    assert sae["basis"]["sha256"] == basis.meta["sha256"] and sae["basis"]["status"] == "verified"

    result = evaluate_basis(basis, source, split, layer=0, allow_unverified_basis=True)
    one = result.to_dict()
    assert one["provenance"]["basis"]["sha256"] == basis.meta["sha256"]
    assert one["train"]["dead"] == [4, 5] and one["split"]["train"] == list(split.train)


def test_a_number_that_is_not_one_is_null_in_the_file(tmp_path):
    """No variance to explain: explained variance is undefined, and JSON has no NaN."""
    source = MemorySource({0: np.zeros((N_TIMES, N_NODES, WIDTH))})
    result = evaluate_basis(_perfect(), source, _split(), layer=0, allow_unverified_basis=True)
    assert np.isnan(result.test.explained_variance)
    text = save_result(tmp_path / "r.json", result).read_text()
    assert "NaN" not in text and json.loads(text)["test"]["explained_variance"] is None


def test_a_pca_curve_point_is_the_pca_of_the_training_times_only():
    source, split = _source(offset_from=9), _split()
    curve = fidelity_curve(source, split, layer=0, pca_components=[4])
    direct = pca_from_moments(accumulate_moments(source, layer=0, times=list(split.train)), 4)
    held_in = np.vstack([source.load(t, 0) for t in split.train])
    assert isinstance(direct, PCA) and held_in.shape[0] == 8 * N_NODES
    ev = curve.method("pca")[0].evaluation
    assert ev.train.explained_variance == pytest.approx(1.0, abs=1e-6)  # rank four, fitted here
    # held out, the offset along a fifth direction cannot be rebuilt by four components
    assert ev.test.explained_variance < 0.95


# -- the command ------------------------------------------------------------------------


def test_the_command_lists_the_split_then_scores_what_was_fitted_on_it(tmp_path):
    from click.testing import CliRunner

    from xaig._cli import cli

    run = CliRunner()
    archive = str(tmp_path / "toy")
    assert run.invoke(cli, ["latents", "toy", archive]).exit_code == 0
    base = ["latents", "evaluate", archive, "--blocks", "4"]

    listed = run.invoke(cli, [*base, "--split-only", "--json"])
    split = json.loads(listed.output)["split"]
    assert len(split["train"]) == 5 and len(split["test"]) == 2 and len(split["buffer"]) == 1

    def fit(name, positions):
        times = [x for i in positions for x in ("--time", str(i))]
        out = str(tmp_path / name)
        done = run.invoke(
            cli, ["latents", "pca", archive, "--components", "4", *times, "--out", out]
        )
        assert done.exit_code == 0, done.output
        return out

    honest = run.invoke(
        cli, [*base, "--basis", fit("a.npz", range(5)), "--pca", "1,4", "--json",
              "--out", str(tmp_path / "r.json")]
    )  # fmt: skip
    assert honest.exit_code == 0, honest.output
    result = json.loads(honest.output)
    assert json.loads((tmp_path / "r.json").read_text()) == result
    assert result["evaluations"][0]["fitted_on"] == "train"
    assert result["evaluations"][0]["provenance"]["basis"]["status"] == "verified"
    assert [p["label"] for p in result["curve"]["points"]] == ["pca-1", "pca-4"]
    assert result["curve"]["points"][1]["explained_variance"] > 0.9

    text = run.invoke(cli, [*base, "--basis", str(tmp_path / "a.npz"), "--pca", "2"])
    assert text.exit_code == 0 and "EV_TEST" in text.output and "pca-2" in text.output

    leaky = run.invoke(cli, [*base, "--basis", fit("all.npz", range(8))])
    assert (
        leaky.exit_code == 1 and leaky.output.startswith("Error: ") and "held-out" in leaky.output
    )
    nothing = run.invoke(cli, base)
    assert nothing.exit_code != 0 and "--basis FILE" in nothing.output
    twins = run.invoke(cli, [*base, "--basis", str(tmp_path / "a.npz"), "--basis",
                             str(tmp_path / "a.npz"), "--stability"])  # fmt: skip
    assert twins.exit_code == 0 and "100.0% of matched features" in twins.output


def test_the_command_checks_each_basis_against_its_pin_and_prints_them(tmp_path):
    from click.testing import CliRunner

    from xaig._cli import cli

    run = CliRunner()
    archive = str(tmp_path / "toy")
    assert run.invoke(cli, ["latents", "toy", archive]).exit_code == 0
    paths, hashes = [], []
    for name, components in (("a.npz", "2"), ("b.npz", "3")):
        out = str(tmp_path / name)
        done = run.invoke(
            cli, ["latents", "pca", archive, "--components", components, "--time", "0",
                  "--time", "1", "--out", out]
        )  # fmt: skip
        assert done.exit_code == 0, done.output
        paths.append(out)
        hashes.append(done.output.rsplit("--basis-sha256 ", 1)[1].strip())
    base = ["latents", "evaluate", archive, "--blocks", "4"]
    both = ["--basis", paths[0], "--basis", paths[1]]

    right = run.invoke(
        cli, [*base, *both, "--basis-sha256", hashes[0], "--basis-sha256", hashes[1]]
    )
    assert right.exit_code == 0, right.output
    line = right.output.strip().splitlines()[-1]
    assert line.startswith("reproduce: xaig latents evaluate ")
    for path, digest in zip(paths, hashes, strict=True):
        assert f"--basis {path} --basis-sha256 {digest}" in line

    swapped = run.invoke(
        cli, [*base, *both, "--basis-sha256", hashes[1], "--basis-sha256", hashes[0]]
    )
    assert swapped.exit_code == 1 and "this is not the basis" in swapped.output
    short = run.invoke(cli, [*base, *both, "--basis-sha256", hashes[0]])
    assert short.exit_code == 1 and short.output.startswith("Error: 1 --basis-sha256 for 2")
    # unpinned still works, and the line it prints pins what it scored
    unpinned = run.invoke(cli, [*base, *both])
    assert unpinned.exit_code == 0 and hashes[1] in unpinned.output
