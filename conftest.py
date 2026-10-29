"""Skips the documentation pages that need sparsity detection where asdex is missing."""

import importlib.util

collect_ignore_glob = (
    [] if importlib.util.find_spec("asdex") else ["docs/guide/derivatives.md"]
)
