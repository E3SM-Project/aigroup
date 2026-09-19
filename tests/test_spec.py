from __future__ import annotations

import pytest
import yaml

from conftest import TOY_SPEC
from xaig.caig import spec as spec_module
from xaig.core.errors import SpecError


@pytest.fixture
def toy():
    return spec_module.from_mapping(yaml.safe_load(TOY_SPEC))


def test_parses_a_grammar_unlike_aug26(toy):
    attrs = toy.parse_id("run001-fast-d1-w08-s1")
    assert attrs["mode"] == "fast"
    assert attrs["depth"] == 1
    assert attrs["width"] == 8
    assert attrs["seed"] == 1


def test_digit_groups_become_ints_but_labels_stay_strings(toy):
    attrs = toy.parse_id("run003-slow-d1-w16-s2")
    assert attrs["num"] == 3
    assert attrs["mode"] == "slow"


@pytest.mark.parametrize(
    "run_id", ["run001-fast-d1-w08-s1", "run002-fast-d2-w08-s1", "run003-slow-d1-w16-s2"]
)
def test_round_trip(toy, run_id):
    assert toy.format_id(toy.parse_id(run_id)) == run_id


def test_zero_padding_is_restored(toy):
    """width=2 means w08, not w8 -- the id must come back byte-identical."""
    assert toy.format_id(toy.parse_id("run001-fast-d1-w08-s1")).endswith("w08-s1")


def test_unknown_factor_key_is_rejected(toy):
    with pytest.raises(SpecError, match="unknown factor key"):
        toy.parse_id("run001-fast-d1-z09-s1")


def test_missing_factor_is_rejected(toy):
    with pytest.raises(SpecError, match="missing width"):
        toy.parse_id("run001-fast-d1-s1")


def test_unmatched_id_is_rejected(toy):
    with pytest.raises(SpecError, match="does not match"):
        toy.parse_id("something-else")


def test_spec_without_id_pattern_yields_no_attributes():
    """An opaque run id is a legitimate design, not an error."""
    spec = spec_module.from_mapping({"name": "opaque"})
    assert spec.parse_id("7f3a9c21") == {}


def test_format_without_template_is_an_error():
    spec = spec_module.from_mapping({"name": "opaque"})
    with pytest.raises(SpecError, match="no id_template"):
        spec.format_id({})


def test_parent_lookup_uses_the_spec_declared_key(toy):
    """toy keys parents by 'num', not 'exp'."""
    assert toy.parent_of({"num": 2}) == "1"
    assert toy.parent_of({"num": 1}) is None


def test_parent_lookup_survives_yaml_integer_keys(toy):
    """An unquoted `2:` in YAML is an int; lookup must still find it."""
    assert toy.parent_of("2") == "1"


def test_missing_name_is_an_error():
    with pytest.raises(SpecError, match="no 'name'"):
        spec_module.from_mapping({"description": "nameless"})


def test_unknown_bundled_spec_lists_what_exists():
    with pytest.raises(SpecError, match="bundled specs"):
        spec_module.load("no-such-campaign")


def test_unknown_key_is_an_error_not_a_silent_default():
    """`id_patern:` must not quietly mean "this campaign has no grammar"."""
    with pytest.raises(SpecError, match="unknown key.*id_patern"):
        spec_module.from_mapping({"name": "typo", "id_patern": "^x$"})


def test_pattern_that_does_not_compile_is_a_spec_error():
    with pytest.raises(SpecError, match="does not compile"):
        spec_module.from_mapping({"name": "bad", "id_pattern": "^(unclosed"})


def test_factor_field_must_be_a_group_of_the_pattern():
    with pytest.raises(SpecError, match="not a named group"):
        spec_module.from_mapping(
            {"name": "bad", "id_pattern": r"^(?P<a>\d+)$", "factor_field": "knobs"}
        )


def test_parents_need_an_explicit_parent_key():
    """No default: 'exp' is one campaign's vocabulary, not a property of campaigns."""
    with pytest.raises(SpecError, match="parent_key"):
        spec_module.from_mapping({"name": "bad", "parents": {"E03": "E02"}})


def test_a_group_may_not_be_called_id_or_status():
    with pytest.raises(SpecError, match="status cannot name"):
        spec_module.from_mapping({"name": "bad", "id_pattern": r"^(?P<status>\w+)$"})


def test_id_fields_follow_the_id_and_skip_the_factor_word(toy):
    assert toy.id_fields == ("num", "mode", "seed")


def test_unformattable_attributes_are_a_spec_error(toy):
    attrs = {**toy.parse_id("run001-fast-d1-w08-s1"), "num": "not-a-number"}
    with pytest.raises(SpecError, match="cannot format"):
        toy.format_id(attrs)


def test_a_directory_named_like_a_spec_does_not_shadow_it(tmp_path, monkeypatch):
    (tmp_path / "aug26").mkdir()
    monkeypatch.chdir(tmp_path)
    assert spec_module.load("aug26").name == "aug26"


# -- the shipped aug26 spec -------------------------------------------------


def test_aug26_is_bundled():
    assert "aug26" in spec_module.bundled_names()


def test_aug26_round_trips_a_real_id():
    spec = spec_module.load("aug26")
    run_id = "E05.aug26.atm.A3_B16_C1_L0_O5_W0_X0.S01"
    attrs = spec.parse_id(run_id)
    assert attrs["exp"] == "E05"
    assert attrs["realm"] == "atm"
    assert attrs["aerosol"] == 3
    assert attrs["batch"] == 16
    assert spec.format_id(attrs) == run_id


def test_aug26_batch_keeps_two_digits():
    spec = spec_module.load("aug26")
    run_id = "E11.aug26.ocn.A0_B08_C0_L0_O5_W0_X0.S01"
    assert spec.format_id(spec.parse_id(run_id)) == run_id


def test_aug26_parent_map():
    spec = spec_module.load("aug26")
    assert spec.parent_of({"exp": "E07"}) == "E05"
    assert spec.parent_of({"exp": "E17"}) == "E11"
    assert spec.parent_of({"exp": "E01"}) is None


# -- factor keys ------------------------------------------------------------


def _keyed(*keys):
    factors = [{"key": k, "name": f"f{i}", "width": 2} for i, k in enumerate(keys)]
    return spec_module.from_mapping(
        {
            "name": "keys",
            "id_pattern": r"^(?P<word>[A-Za-z0-9_]+)$",
            "id_template": "{word}",
            "factor_field": "word",
            "factors": factors,
        }
    )


def test_a_factor_key_may_be_more_than_one_letter():
    """It used to be read as the token's first character, so `LR` loaded without
    complaint and then matched nothing."""
    spec = _keyed("LR", "L", "wd")
    attrs = spec.parse_id("LR03_L01_wd12")
    assert (attrs["f0"], attrs["f1"], attrs["f2"]) == (3, 1, 12)
    assert spec.format_id(attrs) == "LR03_L01_wd12"
    with pytest.raises(SpecError, match="unknown factor key 'LX'"):
        spec.parse_id("LX03_L01_wd12")


def test_factor_keys_that_cannot_work_are_refused_when_the_spec_is_read():
    with pytest.raises(SpecError, match=r"factor key\(s\) A appear twice"):
        _keyed("A", "A")
    with pytest.raises(SpecError, match="must be letters only"):
        _keyed("A1")
    with pytest.raises(SpecError, match=r"factor name\(s\) depth appear twice"):
        spec_module.from_mapping(
            {"name": "n", "factors": [{"key": "a", "name": "depth"}, {"key": "b", "name": "depth"}]}
        )
