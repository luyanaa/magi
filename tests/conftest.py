"""Shared pytest bootstrap: make the repo parent importable as
``brain_moe_pinn`` so test modules can use plain package imports."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
