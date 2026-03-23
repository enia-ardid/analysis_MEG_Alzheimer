"""Helper package for manuscript-facing figure and table builders."""

import os
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_SCRIPT_ROOT / ".mplconfig"))
