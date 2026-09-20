"""Adapters: every assumption about a specific framework lives here.

Adapters are resolved through the ``xaig.adapters`` entry-point group, so a new
system ships as a new module (or a separate distribution) and core never changes.
Heavy dependencies belong here, behind an extra -- never in the base tier.
"""
