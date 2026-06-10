#!/usr/bin/env python3
"""DF²M utility modules.

Also registers shared utility modules from the project root (utils/)
so that imports like ``from utils.load_config import load_config`` work
when dfm_approach/ is the primary package root.
"""

import importlib.util
import sys
from pathlib import Path

_root_utils = Path(__file__).resolve().parents[2] / 'utils'

# Shared modules that live in root utils/ but are imported by dfm_approach code
_SHARED_MODULES = ('load_config', 'handcrafted_features', 'ema', 'losses', 'dataset')

for _name in _SHARED_MODULES:
    _key = f'utils.{_name}'
    _path = _root_utils / f'{_name}.py'
    if _key not in sys.modules and _path.exists():
        _spec = importlib.util.spec_from_file_location(_key, str(_path))
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_key] = _mod
        _spec.loader.exec_module(_mod)
