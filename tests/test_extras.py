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
        # `pip install 'xaig[heavy]'` gets wrong: there is no index to fetch it from.
        (
            {"url": "file:///home/me/ai%20group", "dir_info": {"editable": True}},
            "uv pip install -e '/home/me/ai group[heavy]'",
        ),
        ({"url": "file:///srv/xaig", "dir_info": {}}, "uv pip install '/srv/xaig[heavy]'"),
        (
            {"url": "https://github.com/E3SM-Project/aigroup", "vcs_info": {"vcs": "git"}},
            "uv pip install 'xaig[heavy] @ git+https://github.com/E3SM-Project/aigroup'",
        ),
        # Pinned to a commit, from a branch that has moved since: adding an extra must
        # not quietly change the code that is installed.
        (
            {
                "url": "https://github.com/E3SM-Project/aigroup",
                "vcs_info": {"vcs": "git", "requested_revision": "main", "commit_id": "656212e"},
            },
            "uv pip install 'xaig[heavy] @ git+https://github.com/E3SM-Project/aigroup@656212e'",
        ),
        (
            {
                "url": "https://example.org/mono",
                "subdirectory": "python/xaig",
                "vcs_info": {"vcs": "git", "commit_id": "abc123"},
            },
            "uv pip install 'xaig[heavy] @ git+https://example.org/mono@abc123#subdirectory=python/xaig'",
        ),
        (
            {"url": "https://example.org/xaig-0.1.0.tar.gz", "archive_info": {}},
            "uv pip install 'xaig[heavy] @ https://example.org/xaig-0.1.0.tar.gz'",
        ),
        (None, "uv pip install 'xaig[heavy]'"),  # from an index
        (
            # a wheel on disk is not a checkout: no `uv sync` to offer
            {"url": "file:///tmp/xaig-0.1.0-py3-none-any.whl", "archive_info": {}},
            "uv pip install '/tmp/xaig-0.1.0-py3-none-any.whl[heavy]'",
        ),
    ],
)
def test_the_hint_follows_where_xaig_came_from(monkeypatch, direct_url, expected):
    monkeypatch.setattr(extras, "distribution", lambda name: _Distribution(direct_url))
    assert extras.install_hint("heavy").startswith(expected)


def test_require_names_the_module_the_extra_and_the_command():
    with pytest.raises(MissingExtraError, match="no_such_module .* 'heavy' extra: uv pip install"):
        extras.require("no_such_module", "heavy")


def test_a_wheel_on_disk_is_not_called_a_checkout(monkeypatch):
    origin = {"url": "file:///tmp/xaig-0.1.0-py3-none-any.whl", "archive_info": {}}
    monkeypatch.setattr(extras, "distribution", lambda name: _Distribution(origin))
    assert "uv sync" not in extras.install_hint("heavy")


def test_a_missing_extra_is_still_an_import_error():
    """So `except ImportError` around an optional import keeps working."""
    assert issubclass(MissingExtraError, ImportError)


def test_a_checkout_is_told_the_sync_that_brings_this_extra(monkeypatch):
    """Not "`uv sync` brings all": the default groups need not hold every extra."""
    origin = {"url": "file:///srv/xaig", "dir_info": {"editable": True}}
    monkeypatch.setattr(extras, "distribution", lambda name: _Distribution(origin))
    assert extras.install_hint("heavy").endswith("(in that checkout: `uv sync --extra heavy`)")
