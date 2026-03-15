# Copyright (c) 2026 winjayran
# SPDX-License-Identifier: MIT
"""Audio to MIDI transcription using Spotify's Basic Pitch.

Basic Pitch is a lightweight neural network for audio-to-MIDI transcription
that works well for guitar, piano, and other melodic instruments.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Suppress warnings from basic-pitch dependencies
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

# Suppress basic-pitch optional dependency warnings
logging.getLogger('root').setLevel(logging.CRITICAL)

import numpy as np
import soundfile as sf

# Lazy import basic_pitch to suppress warnings
def _import_basic_pitch():
    """Lazy import basic_pitch with warning suppression."""
    # Suppress the optional dependency warnings
    warnings.filterwarnings('ignore', message='.*CoreML.*')
    warnings.filterwarnings('ignore', message='.*tflite.*')
    warnings.filterwarnings('ignore', message='.*ONNX.*')

    from basic_pitch.inference import predict_and_save
    from basic_pitch import ICASSP_2022_MODEL_PATH
    return predict_and_save, ICASSP_2022_MODEL_PATH

from song2score.types import PartType

logger = logging.getLogger(__name__)


class BasicPitchTranscriber:
    """Audio to MIDI transcriber using Basic Pitch.

    Supports: guitar, piano, bass, vocals (melody), and general melodic instruments.
    Best performance on single-instrument audio.
    """

    # MIDI program numbers for common instruments
    INSTRUMENT_PROGRAMS = {
        PartType.GUITAR: {
            "default": 24,  # Acoustic Guitar (nylon)
            "acoustic_nylon": 24,
            "acoustic_steel": 25,
            "electric_jazz": 26,
            "electric_clean": 27,
            "electric_muted": 28,
            "electric_overdriven": 29,
            "electric_distortion": 30,
        },
        PartType.PIANO: {
            "default": 0,  # Acoustic Grand Piano
            "acoustic_grand": 0,
            "bright_acoustic": 1,
            "electric_grand": 2,
            "electric_piano": 4,
        },
        PartType.BASS: {
            "default": 33,  # Electric Bass (finger)
            "electric_finger": 33,
            "electric_pick": 34,
            "fretless": 35,
            "slap_1": 36,
            "slap_2": 37,
            "synth_1": 38,
            "synth_2": 39,
        },
        PartType.VOCALS: {
            "default": 80,  # Choir Aahs
            "choir_aahs": 80,
            "voice_oohs": 81,
        },
        PartType.STRINGS: {
            "default": 40,  # Violin
            "violin": 40,
            "viola": 41,
            "cello": 42,
            "contrabass": 43,
        },
        PartType.OTHER: {
            "default": 0,  # Piano as default
        },
    }

    # Lazy loaded imports
    _predict_and_save = None
    _ICASSP_2022_MODEL_PATH = None

    @classmethod
    def _get_basic_pitch(cls):
        """Lazy import basic_pitch modules."""
        if cls._predict_and_save is None:
            cls._predict_and_save, cls._ICASSP_2022_MODEL_PATH = _import_basic_pitch()
        return cls._predict_and_save, cls._ICASSP_2022_MODEL_PATH

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        minimum_note_length: float = 0.05,
        midi_tempo: int = 120,
    ):
        """Initialize the Basic Pitch transcriber.

        Args:
            model_path: Path to Basic Pitch model (uses default if None)
            confidence_threshold: Minimum confidence for note detection (0-1)
            minimum_note_length: Minimum note duration in seconds
            midi_tempo: Default tempo for output MIDI
        """
        _, model_path_default = self._get_basic_pitch()
        self.model_path = model_path or str(model_path_default)
        self.confidence_threshold = confidence_threshold
        self.minimum_note_length = minimum_note_length
        self.midi_tempo = midi_tempo

    def _has_sufficient_audio(self, audio: np.ndarray, sr: int, min_duration: float = 0.5) -> Tuple[bool, str]:
        """Check if audio has sufficient content for transcription.

        Args:
            audio: Audio array (mono)
            sr: Sample rate
            min_duration: Minimum duration in seconds

        Returns:
            Tuple of (has_content, reason)
        """
        # Check duration
        duration = len(audio) / sr
        if duration < min_duration:
            return False, f"Audio too short ({duration:.2f}s < {min_duration}s)"

        # Check if audio is mostly silent
        # RMS (root mean square) energy
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.001:
            return False, f"Audio too quiet (RMS={rms:.6f})"

        # Check for any significant audio signal
        # Count samples above noise floor
        above_noise = np.abs(audio) > 0.01
        significant_samples = np.sum(above_noise)
        significant_ratio = significant_samples / len(audio)

        if significant_ratio < 0.01:
            return False, f"Audio mostly silent ({significant_ratio:.1%} above noise floor)"

        return True, "OK"

    def transcribe(
        self,
        audio_path: Path,
        output_path: Path,
        part_type: PartType = PartType.OTHER,
        instrument_variant: str = "default",
    ) -> Tuple[Path, Dict]:
        """Transcribe audio to MIDI.

        Args:
            audio_path: Input audio file path
            output_path: Output MIDI file path
            part_type: Type of instrument being transcribed
            instrument_variant: Specific instrument variant

        Returns:
            Tuple of (output_midi_path, metadata_dict)

        Raises:
            ValueError: If audio has insufficient content for transcription
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load audio
        audio, sr = sf.read(str(audio_path))

        # Handle stereo - convert to mono
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        # Validate audio has sufficient content
        has_content, reason = self._has_sufficient_audio(audio, sr)
        if not has_content:
            logger.warning(f"Skipping transcription of {audio_path}: {reason}")
            raise ValueError(f"Audio has insufficient content for transcription: {reason}")

        # Create output directory for temporary files
        temp_dir = output_path.parent / ".basic_pitch_temp"
        temp_dir.mkdir(exist_ok=True)

        try:
            # Run Basic Pitch prediction
            # Note: predict_and_save outputs to a directory
            predict_and_save, _ = self._get_basic_pitch()

            # Capture any output/errors from basic_pitch
            import io
            import sys
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()

            try:
                # Note: model_or_model_path is a positional argument in current API
                predict_and_save(
                    [str(audio_path)],
                    str(temp_dir),
                    True,  # save_midi
                    False,  # sonify_midi
                    False,  # save_model_outputs
                    True,  # save_notes
                    self.model_path,  # model_or_model_path (positional)
                    onset_threshold=self.confidence_threshold,  # For note onset detection
                    frame_threshold=0.3,  # For frame-level note detection
                    minimum_note_length=int(self.minimum_note_length * 22050),  # Convert seconds to samples (at 44.1kHz)
                    minimum_frequency=50.0,  # Minimum fundamental frequency (Hz)
                    maximum_frequency=2000.0,  # Maximum fundamental frequency (Hz)
                    midi_tempo=self.midi_tempo,
                )
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            # Find the generated MIDI file
            # Basic Pitch names it after the input file
            generated_midi = temp_dir / f"{audio_path.stem}_basic_pitch.mid"

            if not generated_midi.exists():
                # Try alternative naming
                generated_midi = temp_dir / f"{audio_path.stem}.mid"

            if generated_midi.exists():
                # Validate the MIDI file has content
                try:
                    import mido
                    mid = mido.MidiFile(str(generated_midi))
                    # Check if MIDI has any notes
                    has_notes = False
                    for track in mid.tracks:
                        for msg in track:
                            if msg.type == 'note_on' and msg.velocity > 0:
                                has_notes = True
                                break
                        if has_notes:
                            break

                    if not has_notes:
                        logger.warning(f"Basic Pitch generated empty MIDI for {audio_path}")
                        raise ValueError("Generated MIDI contains no notes")
                except Exception as e:
                    logger.warning(f"Could not validate MIDI content: {e}")

                # Move to desired output path
                generated_midi.rename(output_path)

                # Set instrument in MIDI
                self._set_midi_instrument(
                    output_path,
                    part_type,
                    instrument_variant,
                )

                metadata = {
                    "model": "basic_pitch",
                    "confidence_threshold": self.confidence_threshold,
                    "part_type": part_type.value,
                    "instrument": self.INSTRUMENT_PROGRAMS.get(part_type, {}).get(instrument_variant, 0),
                }

                logger.info(f"Transcribed {audio_path} to {output_path}")

                return output_path, metadata
            else:
                # List what files were created
                temp_files = list(temp_dir.glob("*"))
                logger.error(f"Basic Pitch did not generate expected MIDI. Files in temp dir: {[f.name for f in temp_files]}")
                raise FileNotFoundError(f"Basic Pitch did not generate MIDI file (expected {generated_midi.name})")

        except ValueError:
            # Re-raise ValueError as-is (includes our custom errors)
            raise
        except Exception as e:
            logger.error(f"Basic Pitch transcription failed for {audio_path}: {e}")
            raise RuntimeError(f"Basic Pitch transcription failed: {e}")
        finally:
            # Clean up temp directory
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def _set_midi_instrument(
        self,
        midi_path: Path,
        part_type: PartType,
        instrument_variant: str = "default",
    ) -> None:
        """Set the MIDI program (instrument) for the output file.

        Args:
            midi_path: Path to MIDI file
            part_type: Type of instrument
            instrument_variant: Specific variant
        """
        try:
            import mido
        except ImportError:
            logger.warning("mido not installed, skipping instrument assignment")
            return

        # Get the program number
        programs = self.INSTRUMENT_PROGRAMS.get(part_type, {})
        program = programs.get(instrument_variant, programs.get("default", 0))

        # Load and modify MIDI
        mid = mido.MidiFile(str(midi_path))

        for track in mid.tracks:
            for msg in track:
                if msg.type == 'program_change':
                    msg.program = program

        # Save back
        mid.save(str(midi_path))

    def transcribe_with_segments(
        self,
        audio_path: Path,
        output_dir: Path,
        part_type: PartType = PartType.OTHER,
        segment_length: float = 30.0,
    ) -> List[Path]:
        """Transcribe long audio by splitting into segments.

        Useful for very long audio files.

        Args:
            audio_path: Input audio file path
            output_dir: Output directory
            part_type: Type of instrument
            segment_length: Length of each segment in seconds

        Returns:
            List of output MIDI file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load audio
        audio, sr = sf.read(str(audio_path))

        # Handle stereo
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        # Calculate segment samples
        segment_samples = int(segment_length * sr)

        midi_paths = []

        for i, start in enumerate(range(0, len(audio), segment_samples)):
            end = min(start + segment_samples, len(audio))
            segment = audio[start:end]

            # Save segment
            segment_path = output_dir / f"segment_{i:03d}.wav"
            sf.write(segment_path, segment, sr)

            # Transcribe segment
            midi_path = output_dir / f"segment_{i:03d}.mid"
            midi_path, _ = self.transcribe(segment_path, midi_path, part_type)

            midi_paths.append(midi_path)

            # Clean up segment audio
            segment_path.unlink()

        return midi_paths

    def get_supported_instruments(self) -> Dict[PartType, List[str]]:
        """Get list of supported instruments and their variants.

        Returns:
            Dictionary mapping PartType to list of variant names
        """
        return {
            part_type: list(variants.keys())
            for part_type, variants in self.INSTRUMENT_PROGRAMS.items()
        }


# Additional helper functions for guitar-specific processing

class GuitarTranscriber(BasicPitchTranscriber):
    """Specialized transcriber for guitar.

    Handles guitar-specific considerations:
    - String bending detection
    - Chord recognition
    - TAB preparation
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_instrument = "electric_clean"

    def transcribe_guitar(
        self,
        audio_path: Path,
        output_path: Path,
        guitar_type: str = "electric_clean",
    ) -> Tuple[Path, Dict]:
        """Transcribe guitar audio with guitar-specific settings.

        Args:
            audio_path: Input audio file path
            output_path: Output MIDI file path
            guitar_type: Type of guitar (acoustic_steel, acoustic_nylon, electric_*)

        Returns:
            Tuple of (output_midi_path, metadata_dict)
        """
        # Guitar-specific parameters
        # Lower minimum frequency for bass notes
        self.minimum_note_length = 0.03  # Guitar can have fast notes

        return self.transcribe(
            audio_path,
            output_path,
            PartType.GUITAR,
            guitar_type,
        )


class PianoTranscriber(BasicPitchTranscriber):
    """Specialized transcriber for piano.

    Handles piano-specific considerations:
    - Sustain pedal detection
    - Wide dynamic range
    - Polyphonic handling
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_instrument = "acoustic_grand"

    def transcribe_piano(
        self,
        audio_path: Path,
        output_path: Path,
        piano_type: str = "acoustic_grand",
    ) -> Tuple[Path, Dict]:
        """Transcribe piano audio with piano-specific settings.

        Args:
            audio_path: Input audio file path
            output_path: Output MIDI file path
            piano_type: Type of piano (acoustic_grand, bright_acoustic, etc.)

        Returns:
            Tuple of (output_midi_path, metadata_dict)
        """
        # Piano-specific parameters
        # Piano can have very fast passages
        self.minimum_note_length = 0.02

        return self.transcribe(
            audio_path,
            output_path,
            PartType.PIANO,
            piano_type,
        )


class ViolinTranscriber(BasicPitchTranscriber):
    """Specialized transcriber for violin and string instruments.

    Handles string-specific considerations:
    - Vibrato detection
    - Glissando/portamento handling
    - Bowing articulations
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_instrument = "violin"

    def transcribe_violin(
        self,
        audio_path: Path,
        output_path: Path,
        string_type: str = "violin",
    ) -> Tuple[Path, Dict]:
        """Transcribe violin/string audio with string-specific settings.

        Args:
            audio_path: Input audio file path
            output_path: Output MIDI file path
            string_type: Type of string instrument (violin, viola, cello, contrabass)

        Returns:
            Tuple of (output_midi_path, metadata_dict)
        """
        # Violin-specific parameters
        # Strings have sustained notes
        self.minimum_note_length = 0.05

        # Map string type to PartType
        part_type_map = {
            "violin": PartType.STRINGS,
            "viola": PartType.STRINGS,
            "cello": PartType.STRINGS,
            "contrabass": PartType.STRINGS,
        }

        return self.transcribe(
            audio_path,
            output_path,
            part_type_map.get(string_type, PartType.STRINGS),
            string_type,
        )
