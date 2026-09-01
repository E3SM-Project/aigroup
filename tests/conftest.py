"""Synthetic fixtures only.

Nothing here touches a real campaign or a real filesystem outside tmp_path: the
suite has to pass on a laptop with no scratch mounted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Deliberately unlike aug26 -- different separator, different field order, a
# two-letter factor key, no realm. If core is quietly shaped by one campaign,
# this fixture is what fails.
TOY_SPEC = r"""
name: toy
description: A tiny campaign that looks nothing like aug26.
id_pattern: '^run(?P<num>\d+)-(?P<mode>fast|slow)-(?P<knobs>[a-z0-9-]+)-s(?P<seed>\d+)$'
id_template: 'run{num:03d}-{mode}-{knobs}-s{seed:d}'
factor_field: knobs
factor_separator: "-"
factors:
  - { key: d, name: depth, width: 1 }
  - { key: w, name: width, width: 2 }
parent_key: num
parents:
  2: 1
metric: loss
noise_floor: 0.01
discovery:
  adapter: table
  id_column: name
"""

TOY_TABLE = """name,mode,note
run001-fast-d1-w08-s1,fast,baseline
run002-fast-d2-w08-s1,fast,deeper
run003-slow-d1-w16-s2,slow,wider and slower
"""


@pytest.fixture
def toy_spec_path(tmp_path: Path) -> Path:
    path = tmp_path / "toy.yaml"
    path.write_text(TOY_SPEC)
    return path


@pytest.fixture
def toy_table_path(tmp_path: Path) -> Path:
    path = tmp_path / "runs.csv"
    path.write_text(TOY_TABLE)
    return path
