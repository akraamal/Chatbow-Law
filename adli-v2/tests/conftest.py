"""Configuration pytest pour les tests v2 : rend `adli_v2` (dans adli-v2/)
et les modules v1 (src, scripts, app) importables depuis la racine du dépôt."""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_V2_ROOT = _HERE.parent      # adli-v2/
_REPO_ROOT = _V2_ROOT.parent  # racine du dépôt

for _p in (str(_REPO_ROOT), str(_V2_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)