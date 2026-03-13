# Copyright (c) 2026 winjayran
# SPDX-License-Identifier: MIT
"""Audio preprocessing module using ffmpeg and librosa."""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Audio preprocessing using ffmpeg and librosa."""

    # Standard processing parameters
    DEFAULT_SAMPLE_RATE = 44100
    DEFAULT_CHANNELS = 2  # Stereo
    TARGET_SAMPLE_RATE = 44100  # Demucs and Basic Pitch work best at 44.1kHz

    # Supported audio formats (extensions)
    SUPPORTED_FORMATS: List[str] = [
        ".wav", ".wave",      # WAV
        ".mp3", ".mp2",       # MPEG
        ".flac",              # FLAC
        ".ogg", ".oga",       # OGG Vorbis/Opus
        ".m4a", ".mp4", ".aac",  # AAC/MP4
        ".wma",               # Windows Media
        ".aiff", ".aif", ".aifc",  # AIFF
        ".ape",               # Monkey's Audio
        ".wv",                # WavPack
        ".opus",              # Opus
        ".ac3",               # AC-3
    ]

    @classmethod
    def is_supported_format(cls, path: Path) -> bool:
        """Check if a file format is supported.

        Args:
            path: File path to check

        Returns:
            True if format is supported
        """
        return path.suffix.lower() in cls.SUPPORTED_FORMATS

    def __init__(
        self,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        normalize: bool = True,
        mono: bool = False,
    ):
        """Initialize the audio preprocessor.

        Args:
            target_sample_rate: Target sample rate in Hz
            normalize: Whether to normalize audio
            mono: Whether to convert to mono
        """
        self.target_sample_rate = target_sample_rate
        self.normalize = normalize
        self.mono = mono

    def check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def convert_with_ffmpeg(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
    ) -> Path:
        """Convert audio using ffmpeg.

        Args:
            input_path: Input audio file path
            output_path: Output path (if None, creates temp file)
            sample_rate: Target sample rate (uses self.target_sample_rate if None)
            channels: Target channels (1=mono, 2=stereo, None=keep original)

        Returns:
            Path to the converted audio file
        """
        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix=".wav"))

        sample_rate = sample_rate or self.target_sample_rate

        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i", str(input_path),
            "-ar", str(sample_rate),
        ]

        if channels is not None:
            cmd.extend(["-ac", str(channels)])

        cmd.extend(["-acodec", "pcm_s16le", str(output_path)])

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running ffmpeg: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"ffmpeg failed: {result.stderr}")
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")

        # Verify the output file was created
        if not output_path.exists():
            raise RuntimeError(f"ffmpeg did not create output file: {output_path}")

        logger.info(f"ffmpeg conversion successful: {output_path}")

        return output_path

    def load_audio(
        self,
        path: Path,
        sample_rate: Optional[int] = None,
        mono: Optional[bool] = None,
    ) -> Tuple[np.ndarray, int]:
        """Load audio file.

        Args:
            path: Path to audio file
            sample_rate: Target sample rate (uses class default if None)
            mono: Whether to load as mono (uses class default if None)

        Returns:
            Tuple of (audio_array, sample_rate)
            Audio array shape is (samples, channels) for sf.write compatibility
        """
        sample_rate = sample_rate or self.target_sample_rate
        mono = mono if mono is not None else self.mono

        # Use librosa for loading (handles many formats)
        audio, sr = librosa.load(
            path,
            sr=sample_rate,
            mono=mono,
        )

        # Librosa returns (channels, samples) for stereo, but sf.write expects (samples, channels)
        # Transpose if needed
        if audio.ndim == 2 and audio.shape[0] == 2:
            # Shape is (2, samples), transpose to (samples, 2)
            audio = audio.T

        return audio, sr

    def normalize_audio(self, audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        """Normalize audio to target dB level.

        Args:
            audio: Input audio array
            target_db: Target level in dB (usually -3 to -1)

        Returns:
            Normalized audio array
        """
        # Calculate current peak
        peak = np.abs(audio).max()

        if peak == 0:
            return audio

        # Calculate target amplitude
        target_amplitude = 10 ** (target_db / 20)

        # Apply normalization
        normalized = audio * (target_amplitude / peak)

        # Clip to prevent overflow
        return np.clip(normalized, -1.0, 1.0)

    def preprocess(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Full preprocessing pipeline.

        Args:
            input_path: Input audio file path
            output_path: Output path (if None, creates temp file)

        Returns:
            Path to preprocessed audio file
        """
        # First, use ffmpeg to ensure format and sample rate
        temp_path = self.convert_with_ffmpeg(input_path, output_path)

        if self.normalize:
            # Load, normalize, and save
            audio, sr = self.load_audio(temp_path)
            audio = self.normalize_audio(audio)
            sf.write(temp_path, audio, sr)

        return temp_path

    def get_audio_info(self, path: Path) -> dict:
        """Get information about an audio file.

        Args:
            path: Path to audio file

        Returns:
            Dictionary with audio info (duration, sample_rate, channels, etc.)
        """
        info = sf.info(str(path))
        return {
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "format": info.format,
            "subtype": info.subtype,
        }

    def split_audio(
        self,
        audio: np.ndarray,
        segment_length: float,
        sample_rate: int,
    ) -> list[np.ndarray]:
        """Split audio into segments of fixed length.

        Useful for processing long audio files in chunks.

        Args:
            audio: Input audio array
            segment_length: Length of each segment in seconds
            sample_rate: Sample rate

        Returns:
            List of audio segments
        """
        segment_samples = int(segment_length * sample_rate)
        segments = []

        for i in range(0, len(audio), segment_samples):
            segment = audio[i : i + segment_samples]
            segments.append(segment)

        return segments
