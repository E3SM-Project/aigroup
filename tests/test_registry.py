"""The adapter factory contract: ``factory(source, **options)``, validated."""

from __future__ import annotations

import pytest

from xaig.core import registry
from xaig.core.errors import AdapterError


class Reader:
    def __init__(self, url="https://default.example", retries=3, spec=None):
        self.url, self.retries, self.spec = url, retries, spec


class OpenEnded:
    def __init__(self, source, **anything):
        self.source, self.anything = source, anything


class NeedsToken:
    def __init__(self, source, *, token):
        self.token = token


class Sourceless:
    """Reads nothing, so its options are keyword-only: the first positional
    parameter of a factory is always the source."""

    def __init__(self, *, flavour="plain"):
        self.flavour = flavour


@pytest.fixture(autouse=True)
def adapters():
    names = {"reader": Reader, "open": OpenEnded, "token": NeedsToken, "sourceless": Sourceless}
    for name, factory in names.items():
        registry.register(name, factory)
    yield
    for name in names:
        registry.unregister(name)


def test_source_reaches_the_first_parameter_whatever_it_is_called():
    """It used to be passed as `path=`, and dropped when nothing was called that."""
    assert registry.create("reader", source="https://asked-for.example").url.startswith(
        "https://asked"
    )


def test_a_missing_source_falls_back_to_the_factorys_default():
    assert registry.create("reader").url == "https://default.example"


def test_unknown_option_is_an_error_naming_the_accepted_ones():
    with pytest.raises(AdapterError, match=r"option\(s\) retrys; accepted: retries"):
        registry.create("reader", options={"retrys": 5})


def test_context_is_offered_not_forced():
    assert registry.create("reader", context={"spec": "S"}).spec == "S"
    assert registry.create("sourceless", context={"spec": "S"}).flavour == "plain"


def test_context_is_not_listed_as_a_user_option():
    with pytest.raises(AdapterError) as err:
        registry.create("reader", options={"nope": 1}, context={"spec": "S"})
    assert "spec" not in str(err.value).split("accepted:")[1]


def test_an_option_may_not_pose_as_context():
    with pytest.raises(AdapterError, match="set by xaig"):
        registry.create("reader", options={"spec": "mine"}, context={"spec": "S"})


def test_open_ended_factories_take_everything():
    built = registry.create("open", source="s", options={"a": 1}, context={"spec": "S"})
    assert built.anything == {"a": 1, "spec": "S"}


def test_required_source_and_options_are_reported_by_name():
    with pytest.raises(AdapterError, match="needs a source"):
        registry.create("token")
    with pytest.raises(AdapterError, match=r"missing required option\(s\): token"):
        registry.create("token", source="s")


def test_a_source_nobody_would_read_is_an_error():
    with pytest.raises(AdapterError, match="does not take a source"):
        registry.create("sourceless", source="somewhere")


def test_naming_the_source_as_an_option_is_explained():
    with pytest.raises(AdapterError, match="pass it as the source"):
        registry.create("reader", options={"url": "x"})


def test_shipped_adapters_are_found_through_entry_points_alone():
    assert "table" in registry.available()
    assert registry.get("table").__name__ == "TableDiscoverer"


def test_unknown_adapter_names_the_alternatives():
    with pytest.raises(AdapterError, match="available: .*table"):
        registry.get("no-such-adapter")
