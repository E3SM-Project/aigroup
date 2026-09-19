"""A missing extra must say how to get it *for this installation*."""

from __future__ import annotations

import json

import pytest

from xaig.core import extras
from xaig.core.errors import MissingExtraError


class _Distribution:
    def __init__(self, direct_url):
        self._direct_url = direct_url

    def read_text(self, name):
        return None if self._direct_url is None else json.dumps(self._direct_url)


@pytest.mark.parametrize(
    ("direct_url", "expected"),
    [
        # A checkout installed editable -- the usual case, and the one a bare
        # `pip install 'xaig[daig]'` gets wrong: there is no index to fetch it from.
        (
            {"url": "file:///home/me/ai%20group", "dir_info": {"editable": True}},
            "uv pip install -e '/home/me/ai group[daig]'",
        ),
        ({"url": "file:///srv/xaig", "dir_info": {}}, "uv pip install '/srv/xaig[daig]'"),
        (
            {"url": "https://github.com/E3SM-Project/aigroup", "vcs_info": {"vcs": "git"}},
            "uv pip install 'xaig[daig] @ git+https://github.com/E3SM-Project/aigroup'",
        ),
        (None, "uv pip install 'xaig[daig]'"),  # from an index
    ],
)
def test_the_hint_follows_where_xaig_came_from(monkeypatch, direct_url, expected):
    monkeypatch.setattr(extras, "distribution", lambda name: _Distribution(direct_url))
    assert extras.install_hint("daig").startswith(expected)


def test_require_names_the_module_the_extra_and_the_command():
    with pytest.raises(MissingExtraError, match="no_such_module .* 'daig' extra: uv pip install"):
        extras.require("no_such_module", "daig")


def test_a_missing_extra_is_still_an_import_error():
    """So `except ImportError` around an optional import keeps working."""
    assert issubclass(MissingExtraError, ImportError)


def test_a_checkout_is_told_the_sync_that_brings_this_extra(monkeypatch):
    """Not "`uv sync` brings all": the default groups leave torch out."""
    origin = {"url": "file:///srv/xaig", "dir_info": {"editable": True}}
    monkeypatch.setattr(extras, "distribution", lambda name: _Distribution(origin))
    assert extras.install_hint("taig").endswith("(in that checkout: `uv sync --extra taig`)")
