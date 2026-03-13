# Copyright (c) 2026 winjayran
# SPDX-License-Identifier: MIT
"""
song2score: Convert mixed audio songs to separated parts, MIDI, and sheet music.
"""

import warnings
import os

# Suppress warnings from transitive dependencies before any imports
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

# Suppress requests/urllib3 version mismatch warnings (from scikit-learn)
warnings.filterwarnings('ignore', message='.*urllib3.*doesn\'t match.*')
warnings.filterwarnings('ignore', message='.*chardet.*doesn\'t match.*')
warnings.filterwarnings('ignore', message='.*charset_normalizer.*doesn\'t match.*')

__version__ = "0.2.0"

# Lazy import Pipeline to avoid loading heavy dependencies at import time
def __getattr__(name):
    """Lazy import Pipeline to avoid loading transitive dependencies."""
    if name == "Pipeline":
        from song2score.pipeline import Pipeline
        return Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Pipeline", "__version__"]
