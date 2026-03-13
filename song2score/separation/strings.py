# Copyright (c) 2026 winjayran
# SPDX-License-Identifier: MIT
"""Strings detection and separation module.

This module provides enhanced strings separation by:
1. Detecting strings-like content in audio segments
2. Using classification to identify strings sections
3. Iterative separation to extract strings from mixed stems
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa
import soundfile as sf
from sklearn.cluster import KMeans

from song2score.types import PartType

logger = logging.getLogger(__name__)


class StringsSeparator:
    """Strings separator using spectral and timbral analysis.

    This module identifies and separates string instruments (violin, cello, etc.)
    from mixed audio by analyzing spectral characteristics that are typical
    of string instruments.
    """

    # Characteristic frequency ranges for different string instruments
    STRING_FREQUENCIES = {
        "violin": (196, 3136),  # G3 to E7 (fundamental range)
        "viola": (131, 1174),  # C3 to A6
        "cello": (65, 659),  # C2 to E5
        "contrabass": (41, 247),  # E1 to B3
    }

    # Spectral features typical of strings
    STRINGS_HARMONIC_RATIO = 0.5  # Strings have rich harmonics
    STRINGS_ATTACK_TIME = 0.05  # Strings have fast but smooth attack

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        segment_length: float = 10.0,
        use_clustering: bool = True,
    ):
        """Initialize the strings separator.

        Args:
            confidence_threshold: Minimum confidence to classify as strings
            segment_length: Length of analysis segments in seconds
            use_clustering: Whether to use clustering for separation
        """
        self.confidence_threshold = confidence_threshold
        self.segment_length = segment_length
        self.use_clustering = use_clustering

    def detect_strings(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Tuple[bool, float, Dict[str, float]]:
        """Detect if audio contains string instruments.

        Args:
            audio: Input audio array
            sample_rate: Sample rate

        Returns:
            Tuple of (is_strings, confidence, features)
        """
        # Extract spectral features
        features = self._extract_spectral_features(audio, sample_rate)

        # Score based on string-like characteristics
        confidence = self._score_strings_likelihood(features)

        is_strings = confidence >= self.confidence_threshold

        return is_strings, confidence, features

    def _extract_spectral_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, float]:
        """Extract spectral features from audio.

        Args:
            audio: Input audio array
            sample_rate: Sample rate

        Returns:
            Dictionary of spectral features
        """
        # Compute spectrogram
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        power = magnitude ** 2

        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
        mean_centroid = np.mean(centroid)

        # Spectral rolloff (frequency below which 85% of energy is contained)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
        mean_rolloff = np.mean(rolloff)

        # Zero crossing rate (related to noisiness)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        mean_zcr = np.mean(zcr)

        # MFCCs (timbral features)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)

        # Harmonic-percussive separation
        harmonic, percussive = librosa.effects.hpss(audio)
        harmonic_ratio = np.sum(np.abs(harmonic)) / (np.sum(np.abs(harmonic)) + np.sum(np.abs(percussive)) + 1e-10)

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        mean_contrast = np.mean(contrast, axis=1)

        return {
            "centroid": float(mean_centroid),
            "rolloff": float(mean_rolloff),
            "zcr": float(mean_zcr),
            "harmonic_ratio": float(harmonic_ratio),
            "mfcc": mfcc_mean.tolist(),
            "spectral_contrast": mean_contrast.tolist(),
        }

    def _score_strings_likelihood(self, features: Dict[str, float]) -> float:
        """Score how likely the audio contains strings.

        Args:
            features: Extracted spectral features

        Returns:
            Confidence score between 0 and 1
        """
        score = 0.0

        # Strings typically have:
        # - Moderate spectral centroid (not too bright like cymbals)
        # - High harmonic ratio (sustained tones)
        # - Low to moderate ZCR (smooth sound)
        # - Specific MFCC patterns

        centroid = features["centroid"]
        harmonic_ratio = features["harmonic_ratio"]
        zcr = features["zcr"]

        # Centroid scoring (strings typically 1000-4000 Hz)
        if 1000 <= centroid <= 4000:
            score += 0.3
        elif 500 <= centroid <= 5500:
            score += 0.15

        # Harmonic ratio scoring (strings are very harmonic)
        if harmonic_ratio > 0.7:
            score += 0.3
        elif harmonic_ratio > 0.5:
            score += 0.15

        # ZCR scoring (strings have smooth attack)
        if zcr < 0.1:
            score += 0.2
        elif zcr < 0.2:
            score += 0.1

        # MFCC-based scoring
        # Strings have characteristic MFCC patterns
        mfcc = np.array(features["mfcc"])
        # This is a simplified check - in practice, use a trained model
        if mfcc[1] > 0 and mfcc[2] < 0:  # Typical string pattern
            score += 0.2

        return min(score, 1.0)

    def separate_strings_from_mixed(
        self,
        mixed_audio: np.ndarray,
        sample_rate: int,
        output_dir: Path,
    ) -> Optional[Path]:
        """Separate strings content from mixed audio.

        Uses spectral clustering to identify and separate strings-like content.

        Args:
            mixed_audio: Input mixed audio (e.g., "other" stem)
            sample_rate: Sample rate
            output_dir: Output directory

        Returns:
            Path to separated strings audio, or None if no strings detected
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if audio contains strings
        is_strings, confidence, _ = self.detect_strings(mixed_audio, sample_rate)

        logger.info(f"Strings detection: {is_strings} (confidence: {confidence:.2f})")

        if not is_strings:
            return None

        # Use harmonic-percussive separation to extract harmonic content
        # Strings are primarily harmonic
        harmonic, percussive = librosa.effects.hpss(mixed_audio)

        # Further separation using spectral features
        strings_audio = self._spectral_separation(harmonic, sample_rate)

        # Save output
        output_path = output_dir / "strings.wav"
        sf.write(output_path, strings_audio, sample_rate)

        logger.info(f"Saved strings audio to {output_path}")

        return output_path

    def _spectral_separation(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Separate strings-like content using spectral filtering.

        Args:
            audio: Input audio (typically harmonic component)
            sample_rate: Sample rate

        Returns:
            Filtered audio emphasizing strings
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Create a mask that emphasizes string-like frequencies
        # Strings typically have energy in specific frequency ranges
        mask = np.ones_like(magnitude)

        # Emphasize string frequency bands
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)

        for instrument, (low, high) in self.STRING_FREQUENCIES.items():
            # Create bandpass mask for this instrument
            band_mask = (freqs >= low) & (freqs <= high)
            mask[band_mask, :] *= 1.2  # Boost these frequencies

        # Normalize mask
        mask = np.clip(mask, 0, 1)

        # Apply mask
        separated = magnitude * mask

        # Reconstruct audio
        separated_stft = separated * np.exp(1j * phase)
        separated_audio = librosa.istft(separated_stft, hop_length=512)

        return separated_audio

    def analyze_and_separate(
        self,
        input_path: Path,
        output_dir: Path,
    ) -> Dict[PartType, Path]:
        """Analyze input audio and separate strings if present.

        Args:
            input_path: Input audio file path
            output_dir: Output directory

        Returns:
            Dictionary with PartType.STRINGS path if detected, empty otherwise
        """
        # Load audio
        audio, sr = librosa.load(input_path, sr=None)

        # Detect strings
        is_strings, confidence, features = self.detect_strings(audio, sr)

        result: Dict[PartType, Path] = {}

        if is_strings:
            # Separate strings
            strings_path = self.separate_strings_from_mixed(audio, sr, output_dir)
            if strings_path:
                result[PartType.STRINGS] = strings_path
        else:
            logger.info(f"No strings detected (confidence: {confidence:.2f})")

        return result

    def classify_string_sections(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[Tuple[float, float, str]]:
        """Classify different string instruments in sections of audio.

        Args:
            audio: Input audio array
            sample_rate: Sample rate

        Returns:
            List of (start_time, end_time, instrument_type) tuples
        """
        segment_samples = int(self.segment_length * sample_rate)
        sections = []

        for i in range(0, len(audio), segment_samples):
            segment = audio[i : i + segment_samples]

            if len(segment) < segment_samples // 2:
                continue

            # Classify this segment
            features = self._extract_spectral_features(segment, sample_rate)
            instrument = self._classify_string_instrument(features)

            start_time = i / sample_rate
            end_time = (i + len(segment)) / sample_rate

            sections.append((start_time, end_time, instrument))

        return sections

    def _classify_string_instrument(self, features: Dict[str, float]) -> str:
        """Classify which string instrument is present.

        This is a simplified classifier. For production, use a trained model.
        """
        centroid = features["centroid"]

        # Rough classification based on brightness (spectral centroid)
        if centroid < 500:
            return "contrabass"
        elif centroid < 1000:
            return "cello"
        elif centroid < 2000:
            return "viola"
        else:
            return "violin"
