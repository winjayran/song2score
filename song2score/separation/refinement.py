# Copyright (c) 2026 winjayran
# SPDX-License-Identifier: MIT
"""Stem refinement module for cleaning up separated audio.

This module provides tools to refine separated stems by removing
unwanted content from other instruments. Uses techniques like:
- Harmonic-Percussive Source Separation (HPSS)
- Frequency band filtering
- Spectral gating
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from song2score.types import PartType

logger = logging.getLogger(__name__)


class StemRefiner:
    """Refine separated stems to remove unwanted content.

    This class implements various techniques to clean up stems that
    contain residual audio from other instruments.
    """

    # Frequency ranges for different instruments (Hz)
    FREQUENCY_RANGES = {
        PartType.BASS: (20, 250),
        PartType.DRUMS: (30, 5000),
        PartType.GUITAR: (80, 5000),
        PartType.PIANO: (28, 4200),
        PartType.STRINGS: (200, 8000),
        PartType.VOCALS: (80, 3500),
        PartType.OTHER: (20, 20000),
    }

    def __init__(
        self,
        use_harmonic_mask: bool = True,
        use_percussive_mask: bool = True,
        use_frequency_filter: bool = True,
        margin: float = 1.0,
    ):
        """Initialize the stem refiner.

        Args:
            use_harmonic_mask: Apply harmonic mask for melodic instruments
            use_percussive_mask: Apply percussive mask for drums
            use_frequency_filter: Apply frequency band filtering
            margin: Margin for HPSS separation (higher = more separation)
        """
        self.use_harmonic_mask = use_harmonic_mask
        self.use_percussive_mask = use_percussive_mask
        self.use_frequency_filter = use_frequency_filter
        self.margin = margin

    def refine_stem(
        self,
        stem_path: Path,
        part_type: PartType,
        output_path: Optional[Path] = None,
    ) -> Tuple[Path, Dict[str, any]]:
        """Refine a single stem file.

        Args:
            stem_path: Path to input stem file
            part_type: Type of instrument in the stem
            output_path: Optional output path (defaults to input path)

        Returns:
            Tuple of (output_path, metadata)
        """
        if output_path is None:
            output_path = stem_path

        # Load audio
        audio, sr = librosa.load(str(stem_path), sr=None, mono=False)

        metadata = {
            "original_shape": audio.shape,
            "sample_rate": sr,
            "part_type": part_type.value,
        }

        # Apply refinements
        refined_audio = audio.copy()

        # 1. Frequency band filtering
        if self.use_frequency_filter:
            refined_audio = self._apply_frequency_filter(
                refined_audio, sr, part_type
            )
            metadata["frequency_filter"] = self.FREQUENCY_RANGES.get(
                part_type, (20, 20000)
            )

        # 2. HPSS-based masking
        if self.use_harmonic_mask or self.use_percussive_mask:
            refined_audio = self._apply_hpss_mask(refined_audio, sr, part_type)
            metadata["hpss_applied"] = True

        # Save refined audio
        sf.write(str(output_path), refined_audio.T, sr)

        metadata["refined_shape"] = refined_audio.shape

        logger.info(f"Refined {part_type.value} stem: {stem_path}")

        return output_path, metadata

    def refine_all_stems(
        self,
        stems: Dict[PartType, Path],
        output_dir: Optional[Path] = None,
    ) -> Dict[PartType, Tuple[Path, Dict[str, any]]]:
        """Refine all stems.

        Args:
            stems: Dictionary of PartType to stem paths
            output_dir: Optional output directory (defaults to overwrite)

        Returns:
            Dictionary of PartType to (output_path, metadata)
        """
        results = {}

        for part_type, stem_path in stems.items():
            if output_dir:
                output_path = output_dir / f"{part_type.value}_refined.wav"
            else:
                # Create temporary path, will replace original
                output_path = stem_path.parent / f"{part_type.value}_temp.wav"

            output_path, metadata = self.refine_stem(stem_path, part_type, output_path)

            # If we created a temp file, replace the original
            if output_dir is None and output_path != stem_path:
                stem_path.unlink()
                output_path.rename(stem_path)
                output_path = stem_path

            results[part_type] = (output_path, metadata)

        return results

    def _apply_frequency_filter(
        self,
        audio: np.ndarray,
        sr: int,
        part_type: PartType,
    ) -> np.ndarray:
        """Apply frequency band filtering based on instrument type.

        Args:
            audio: Input audio (can be mono or stereo)
            sr: Sample rate
            part_type: Instrument type

        Returns:
            Filtered audio
        """
        freq_range = self.FREQUENCY_RANGES.get(part_type)
        if not freq_range:
            return audio

        min_freq, max_freq = freq_range

        # Convert to mono for processing
        if audio.ndim == 2:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio

        # Apply bandpass filter using librosa's mel filterbank approach
        # This is a simplified implementation - for better results, use scipy.signal
        stft = librosa.stft(audio_mono)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Create frequency mask
        freq_bins = librosa.fft_frequencies(sr=sr)
        mask = (freq_bins >= min_freq) & (freq_bins <= max_freq)

        # Apply mask
        filtered_stft = magnitude * mask[:, np.newaxis] * np.exp(1j * phase)
        filtered_audio = librosa.istft(filtered_stft)

        # Restore stereo if needed
        if audio.ndim == 2:
            filtered_audio = np.vstack([filtered_audio, filtered_audio])

        return filtered_audio

    def _apply_hpss_mask(
        self,
        audio: np.ndarray,
        sr: int,
        part_type: PartType,
    ) -> np.ndarray:
        """Apply HPSS-based masking to separate harmonic and percussive content.

        Args:
            audio: Input audio (can be mono or stereo)
            sr: Sample rate
            part_type: Instrument type

        Returns:
            Masked audio
        """
        # Convert to mono for processing
        if audio.ndim == 2:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio

        # Apply HPSS
        harmonic, percussive = librosa.effects.hpss(
            audio_mono,
            margin=self.margin,
        )

        # Select based on instrument type
        if part_type == PartType.DRUMS:
            # Keep percussive, attenuate harmonic
            result = percussive + 0.05 * harmonic
        elif part_type in (PartType.VOCALS, PartType.GUITAR,
                          PartType.PIANO, PartType.STRINGS):
            # Keep harmonic, attenuate percussive
            result = harmonic + 0.05 * percussive
        elif part_type == PartType.BASS:
            # Bass is harmonic but lower frequency
            result = harmonic + 0.02 * percussive
        else:
            # Keep balanced
            result = (harmonic + percussive) / 2

        # Restore stereo if needed
        if audio.ndim == 2:
            result = np.vstack([result, result])

        return result


def refine_vocals_stem(
    stem_path: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """Refine vocals stem to remove drums and other percussive content.

    This is a specialized function for vocals that applies aggressive
    HPSS to remove drum bleed.

    Args:
        stem_path: Path to vocals stem file
        output_path: Optional output path

    Returns:
        Path to refined vocals stem
    """
    refiner = StemRefiner(
        use_harmonic_mask=True,
        use_percussive_mask=False,  # We want to suppress percussive
        use_frequency_filter=True,
        margin=2.0,  # Higher margin for more aggressive separation
    )

    output_path, _ = refiner.refine_stem(
        stem_path,
        PartType.VOCALS,
        output_path,
    )

    return output_path


def refine_bass_stem(
    stem_path: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """Refine bass stem to remove piano, strings, and other high-frequency content.

    Args:
        stem_path: Path to bass stem file
        output_path: Optional output path

    Returns:
        Path to refined bass stem
    """
    refiner = StemRefiner(
        use_harmonic_mask=True,
        use_percussive_mask=False,
        use_frequency_filter=True,
        margin=1.0,
    )

    output_path, _ = refiner.refine_stem(
        stem_path,
        PartType.BASS,
        output_path,
    )

    return output_path
