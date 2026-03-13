# Copyright (c) 2026 winjayran
# SPDX-License-Identifier: MIT
"""
song2score: Convert mixed audio songs to separated parts, MIDI, and sheet music.
"""

__version__ = "0.1.0"

from song2score.pipeline import Pipeline

__all__ = ["Pipeline", "__version__"]
