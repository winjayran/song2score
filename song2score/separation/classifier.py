# Copyright (c) 2026 winjayran
# SPDX-License-Identifier: MIT
"""Instrument classification module using audio feature analysis.

This module provides improved instrument detection by analyzing
various audio features to distinguish between:
- Drums (percussive, transient-rich)
- Guitar (harmonic, plucked string characteristics)
- Piano (harmonic, sustained with decay)
- Strings/violin (harmonic, sustained with vibrato)
- Vocals (formant structure, speech-like patterns)
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa
import soundfile as sf

from song2score.types import PartType

logger = logging.getLogger(__name__)


class InstrumentClass(str, Enum):
    """Instrument classes for classification."""

    DRUMS = "drums"
    GUITAR = "guitar"
    PIANO = "piano"
    STRINGS = "strings"
    VOCALS = "vocals"
    BASS = "bass"
    UNKNOWN = "unknown"


class InstrumentClassifier:
    """Audio instrument classifier using feature-based analysis.

    This classifier uses a combination of spectral features, temporal features,
    and harmonic-percussive separation to identify the dominant instrument
    in an audio segment.
    """

    # Characteristic feature ranges for each instrument type
    # These are heuristic ranges based on common instrument characteristics
    FEATURE_RANGES = {
        InstrumentClass.DRUMS: {
            "harmonic_ratio": (0.0, 0.3),
            "zcr_mean": (0.15, 0.5),
            "spectral_centroid_mean": (100, 2000),
            "attack_time": (0.001, 0.05),
            "decay_slope": (2.0, 10.0),  # Fast decay
        },
        InstrumentClass.GUITAR: {
            "harmonic_ratio": (0.6, 0.9),
            "zcr_mean": (0.05, 0.15),
            "spectral_centroid_mean": (1000, 4000),
            "attack_time": (0.005, 0.03),
            "decay_slope": (0.5, 2.0),
        },
        InstrumentClass.PIANO: {
            "harmonic_ratio": (0.7, 0.95),
            "zcr_mean": (0.02, 0.1),
            "spectral_centroid_mean": (500, 3000),
            "attack_time": (0.005, 0.02),
            "decay_slope": (0.3, 1.5),
        },
        InstrumentClass.STRINGS: {
            "harmonic_ratio": (0.8, 0.98),
            "zcr_mean": (0.02, 0.08),
            "spectral_centroid_mean": (1000, 5000),
            "attack_time": (0.05, 0.2),  # Slower attack
            "decay_slope": (0.1, 0.8),  # Very slow decay
        },
        InstrumentClass.VOCALS: {
            "harmonic_ratio": (0.5, 0.85),
            "zcr_mean": (0.08, 0.2),
            "spectral_centroid_mean": (500, 3500),
            "attack_time": (0.01, 0.1),
            "decay_slope": (0.2, 1.2),
        },
        InstrumentClass.BASS: {
            "harmonic_ratio": (0.6, 0.9),
            "zcr_mean": (0.02, 0.08),
            "spectral_centroid_mean": (50, 400),  # Low frequencies
            "attack_time": (0.005, 0.05),
            "decay_slope": (0.8, 3.0),
        },
    }

    def __init__(
        self,
        segment_length: float = 5.0,
        hop_length: int = 512,
        n_fft: int = 2048,
    ):
        """Initialize the instrument classifier.

        Args:
            segment_length: Length of analysis segments in seconds
            hop_length: Hop length for STFT
            n_fft: FFT size for STFT
        """
        self.segment_length = segment_length
        self.hop_length = hop_length
        self.n_fft = n_fft

    def classify(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Tuple[InstrumentClass, float, Dict[str, float]]:
        """Classify the dominant instrument in audio.

        Args:
            audio: Input audio array (mono or stereo)
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (instrument_class, confidence, features)
        """
        # Convert to mono if stereo
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        # Extract features
        features = self._extract_features(audio, sample_rate)

        # Score each instrument class
        scores = {}
        for instrument_class in InstrumentClass:
            if instrument_class == InstrumentClass.UNKNOWN:
                continue
            scores[instrument_class] = self._score_instrument(
                features, instrument_class
            )

        # Get best match
        best_class = max(scores, key=scores.get)
        confidence = scores[best_class]

        logger.info(f"Classification scores: {scores}")
        logger.info(f"Best match: {best_class} (confidence: {confidence:.2f})")

        return best_class, confidence, features

    def classify_file(
        self,
        audio_path: Path,
    ) -> Tuple[InstrumentClass, float, Dict[str, float]]:
        """Classify the dominant instrument in an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (instrument_class, confidence, features)
        """
        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=None)

        return self.classify(audio, sr)

    def classify_segments(
        self,
        audio_path: Path,
    ) -> List[Tuple[float, float, InstrumentClass, float]]:
        """Classify instruments in segments of an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            List of (start_time, end_time, instrument_class, confidence) tuples
        """
        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=None)

        # Convert to mono if stereo
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        segment_samples = int(self.segment_length * sr)
        results = []

        for i in range(0, len(audio), segment_samples):
            segment = audio[i : i + segment_samples]

            if len(segment) < segment_samples // 4:
                continue

            start_time = i / sr
            end_time = (i + len(segment)) / sr

            instrument_class, confidence, _ = self.classify(segment, sr)
            results.append((start_time, end_time, instrument_class, confidence))

        return results

    def _extract_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, float]:
        """Extract audio features for classification.

        Args:
            audio: Input audio array (mono)
            sample_rate: Sample rate in Hz

        Returns:
            Dictionary of feature names and values
        """
        features = {}

        # Harmonic-percussive separation
        harmonic, percussive = librosa.effects.hpss(audio)

        # Harmonic ratio
        harmonic_energy = np.sum(harmonic ** 2)
        percussive_energy = np.sum(percussive ** 2)
        total_energy = harmonic_energy + percussive_energy + 1e-10
        features["harmonic_ratio"] = float(harmonic_energy / total_energy)

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=2048, hop_length=512)[0]
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, sr=sample_rate, hop_length=512
        )[0]
        features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
        features["spectral_centroid_std"] = float(np.std(spectral_centroids))

        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=sample_rate, hop_length=512
        )[0]
        features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(
            y=audio, sr=sample_rate, hop_length=512
        )
        features["spectral_contrast_mean"] = float(np.mean(contrast))

        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        for i, mfcc in enumerate(mfccs):
            features[f"mfcc_{i}_mean"] = float(np.mean(mfcc))

        # Temporal features - attack and decay
        envelope = self._get_envelope(audio)
        features["attack_time"] = float(self._estimate_attack_time(envelope, sample_rate))
        features["decay_slope"] = float(self._estimate_decay_slope(envelope))

        # Onset detection rate
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=sample_rate, hop_length=512, backtrack=True
        )
        onset_rate = len(onset_frames) / (len(audio) / sample_rate) if len(audio) > 0 else 0
        features["onset_rate"] = float(onset_rate)

        return features

    def _get_envelope(self, audio: np.ndarray) -> np.ndarray:
        """Get the amplitude envelope of audio.

        Args:
            audio: Input audio array

        Returns:
            Envelope array
        """
        # Use RMS energy as envelope
        frame_length = 2048
        hop_length = 512

        rms = librosa.feature.rms(
            y=audio, frame_length=frame_length, hop_length=hop_length
        )[0]

        return rms

    def _estimate_attack_time(
        self,
        envelope: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Estimate the attack time from envelope.

        Args:
            envelope: Amplitude envelope
            sample_rate: Sample rate

        Returns:
            Estimated attack time in seconds
        """
        # Find onset (first time above 10% of max)
        threshold = 0.1 * np.max(envelope)
        above_threshold = envelope > threshold

        if not np.any(above_threshold):
            return 0.0

        onset_idx = np.where(above_threshold)[0][0]

        # Find peak
        peak_idx = np.argmax(envelope)

        # Convert to seconds (hop_length = 512)
        hop_length = 512
        attack_time = (peak_idx - onset_idx) * hop_length / sample_rate

        return max(0.0, attack_time)

    def _estimate_decay_slope(self, envelope: np.ndarray) -> float:
        """Estimate the decay slope from envelope.

        Args:
            envelope: Amplitude envelope

        Returns:
            Decay slope (higher = faster decay)
        """
        # Find peak
        peak_idx = np.argmax(envelope)
        peak_value = envelope[peak_idx]

        # Get decay portion (after peak)
        if peak_idx < len(envelope) - 1:
            decay_portion = envelope[peak_idx:]

            # Normalize
            if peak_value > 0:
                decay_portion = decay_portion / peak_value

            # Fit exponential decay and get rate
            # Simplified: use linear slope in log domain
            log_decay = np.log(decay_portion + 1e-10)
            slope = -np.polyfit(range(len(log_decay)), log_decay, 1)[0]

            return max(0.0, slope)

        return 1.0

    def _score_instrument(
        self,
        features: Dict[str, float],
        instrument_class: InstrumentClass,
    ) -> float:
        """Score how well features match an instrument class.

        Args:
            features: Extracted audio features
            instrument_class: Instrument class to score

        Returns:
            Confidence score between 0 and 1
        """
        ranges = self.FEATURE_RANGES.get(instrument_class, {})

        if not ranges:
            return 0.0

        score = 0.0
        weight_sum = 0.0

        # Weights for different features
        weights = {
            "harmonic_ratio": 1.0,
            "zcr_mean": 0.8,
            "spectral_centroid_mean": 0.8,
            "attack_time": 0.6,
            "decay_slope": 0.6,
        }

        for feature_name, (min_val, max_val) in ranges.items():
            if feature_name not in features:
                continue

            value = features[feature_name]
            weight = weights.get(feature_name, 0.5)

            # Check if value is in range
            if min_val <= value <= max_val:
                # Full score if in range
                feature_score = 1.0
            else:
                # Partial score based on distance from range
                distance = min(abs(value - min_val), abs(value - max_val))
                range_width = max_val - min_val
                feature_score = max(0.0, 1.0 - distance / (range_width * 2))

            score += feature_score * weight
            weight_sum += weight

        if weight_sum > 0:
            return score / weight_sum

        return 0.0

    def map_to_part_type(self, instrument_class: InstrumentClass) -> PartType:
        """Map instrument class to PartType.

        Args:
            instrument_class: Instrument class from classification

        Returns:
            Corresponding PartType
        """
        mapping = {
            InstrumentClass.DRUMS: PartType.DRUMS,
            InstrumentClass.GUITAR: PartType.GUITAR,
            InstrumentClass.PIANO: PartType.PIANO,
            InstrumentClass.STRINGS: PartType.STRINGS,
            InstrumentClass.VOCALS: PartType.VOCALS,
            InstrumentClass.BASS: PartType.BASS,
            InstrumentClass.UNKNOWN: PartType.OTHER,
        }

        return mapping.get(instrument_class, PartType.OTHER)


def classify_stem(
    stem_path: Path,
) -> Tuple[PartType, float]:
    """Quick classification function for a stem file.

    Args:
        stem_path: Path to stem audio file

    Returns:
        Tuple of (PartType, confidence)
    """
    classifier = InstrumentClassifier()
    instrument_class, confidence, _ = classifier.classify_file(stem_path)
    part_type = classifier.map_to_part_type(instrument_class)

    return part_type, confidence
