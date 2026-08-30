"""Pytest bootstrap.

Puts ``src/`` on ``sys.path`` so the sub-packages (perception, world_model,
policy, safety, sim, deployment, utils) are importable *without* an editable
install. After ``pip install -e .`` this is a harmless no-op.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
