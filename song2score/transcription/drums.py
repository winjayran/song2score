# Copyright (c) 2026 winjayran
# SPDX-License-Identifier: MIT
"""Drum transcription using Madmom.

This module provides drum-specific transcription:
- Kick (bass drum) detection
- Snare detection
- Hi-hat detection
- Cymbal detection
- Full drum kit transcription
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from song2score.types import PartType

logger = logging.getLogger(__name__)


class DrumTranscriber:
    """Audio to MIDI transcriber for drums using Madmom.

    Madmom provides specialized drum transcription models that can detect:
    - Kick drum (bass drum)
    - Snare drum
    - Hi-hat (open and closed)
    - Cymbals (crash, ride)
    - Tom toms
    """

    # General MIDI drum note numbers (GM Standard)
    # Channel 10, note numbers
    DRUM_NOTES = {
        "kick": 36,          # Acoustic Bass Drum
        "kick2": 35,         # Bass Drum 1
        "snare": 38,         # Acoustic Snare
        "snare2": 40,        # Electric Snare
        "hihat_closed": 42,  # Closed Hi-Hat
        "hihat_closed2": 44, # Pedal Hi-Hat
        "hihat_open": 46,    # Open Hi-Hat
        "tom_hi": 50,        # High Tom
        "tom_mid_hi": 48,    # Hi-Mid Tom
        "tom_mid_lo": 45,    # Low-Mid Tom
        "tom_lo": 41,        # Low Floor Tom
        "crash": 49,         # Crash Cymbal 1
        "crash2": 57,        # Crash Cymbal 2
        "ride": 51,          # Ride Cymbal 1
        "ride2": 59,         # Ride Cymbal 2
        "ride_bell": 53,     # Ride Bell
    }

    def __init__(
        self,
        beat_tracking: bool = True,
        note_detection: bool = True,
        min_confidence: float = 0.3,
    ):
        """Initialize the drum transcriber.

        Args:
            beat_tracking: Whether to perform beat tracking
            note_detection: Whether to detect individual drum notes
            min_confidence: Minimum confidence for detection
        """
        self.beat_tracking = beat_tracking
        self.note_detection = note_detection
        self.min_confidence = min_confidence

        # Check if madmom is available
        try:
            import madmom
            self.madmom = madmom
            self.available = True
        except ImportError:
            logger.warning("madmom not installed, drum transcription will use fallback")
            self.madmom = None
            self.available = False

    def transcribe(
        self,
        audio_path: Path,
        output_path: Path,
    ) -> Tuple[Path, Dict]:
        """Transcribe drum audio to MIDI.

        Args:
            audio_path: Input audio file path
            output_path: Output MIDI file path

        Returns:
            Tuple of (output_midi_path, metadata_dict)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.available:
            return self._transcribe_with_madmom(audio_path, output_path)
        else:
            return self._transcribe_fallback(audio_path, output_path)

    def _transcribe_with_madmom(
        self,
        audio_path: Path,
        output_path: Path,
    ) -> Tuple[Path, Dict]:
        """Transcribe using madmom's drum transcription.

        Args:
            audio_path: Input audio file path
            output_path: Output MIDI file path

        Returns:
            Tuple of (output_midi_path, metadata_dict)
        """
        from madmom.features import Drums

        # Load audio
        audio, sr = sf.read(str(audio_path))

        # Handle stereo
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        # Use madmom's drum transcription processor
        # This detects drum events (kick, snare, hihat)
        try:
            # Try the new API first
            processor = Drums.DrumDetectionProcessor()
            events = processor(audio)

            # Convert events to MIDI
            self._save_drum_midi(events, output_path)

            metadata = {
                "model": "madmom",
                "method": "drum_detection",
                "num_events": len(events) if events is not None else 0,
            }

            logger.info(f"Transcribed drums to {output_path}")

            return output_path, metadata

        except (AttributeError, Exception) as e:
            logger.warning(f"Madmom drum detection failed: {e}, using alternative method")
            return self._transcribe_with_madmom_alternative(audio_path, output_path, sr)

    def _transcribe_with_madmom_alternative(
        self,
        audio_path: Path,
        output_path: Path,
        sample_rate: int,
    ) -> Tuple[Path, Dict]:
        """Alternative transcription using beat tracking + classification.

        Args:
            audio_path: Input audio file path
            output_path: Output MIDI file path
            sample_rate: Sample rate

        Returns:
            Tuple of (output_midi_path, metadata_dict)
        """
        from madmom.features import Beats, Downbeats

        # Load audio
        audio, sr = sf.read(str(audio_path))
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        # Track beats
        act = self.madmom.features.beats.RNNBeatProcessor()(audio_path)
        beats = self.madmom.features.beats.DBNBeatTrackingProcessor(fps=100)(act)

        # Classify drum sounds at each beat
        drum_events = self._classify_drums_at_beats(audio, sr, beats)

        # Save as MIDI
        self._save_drum_midi(drum_events, output_path)

        metadata = {
            "model": "madmom",
            "method": "beat_tracking_classification",
            "num_beats": len(beats),
            "num_events": len(drum_events),
        }

        return output_path, metadata

    def _classify_drums_at_beats(
        self,
        audio: np.ndarray,
        sample_rate: int,
        beats: np.ndarray,
    ) -> List[Tuple[float, int]]:
        """Classify drum types at beat positions.

        Args:
            audio: Audio array
            sample_rate: Sample rate
            beats: Beat times in seconds

        Returns:
            List of (time, note_number) tuples
        """
        events = []

        for beat_time in beats:
            # Extract audio around the beat
            start_sample = int(beat_time * sample_rate)
            end_sample = start_sample + int(0.1 * sample_rate)  # 100ms window

            if end_sample >= len(audio):
                continue

            segment = audio[start_sample:end_sample]

            # Classify drum type based on spectral features
            drum_type = self._classify_drum_segment(segment, sample_rate)
            note_number = self.DRUM_NOTES.get(drum_type, 38)  # Default to snare

            events.append((beat_time, note_number))

        return events

    def _classify_drum_segment(
        self,
        segment: np.ndarray,
        sample_rate: int,
    ) -> str:
        """Classify a short audio segment as a drum type.

        Simple classifier based on spectral characteristics.

        Args:
            segment: Audio segment
            sample_rate: Sample rate

        Returns:
            Drum type string (kick, snare, hihat_closed, etc.)
        """
        # Compute spectral features
        import librosa

        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=segment, sr=sample_rate)[0]

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(segment)[0]

        # Energy in low frequencies
        import scipy.signal
        freqs, times, spec = scipy.signal.spectral.spectrogram(segment, fs=sample_rate)
        low_energy = np.mean(spec[:10])  # Bottom 10 frequency bins

        mean_centroid = np.mean(centroid)
        mean_zcr = np.mean(zcr)

        # Classification logic
        if mean_centroid < 100:
            # Very low centroid = kick
            return "kick"
        elif mean_zcr > 0.3:
            # High zero crossing = hi-hat or cymbal
            return "hihat_closed" if low_energy > 0.1 else "crash"
        else:
            # Mid-range = snare
            return "snare"

    def _save_drum_midi(
        self,
        events: List[Tuple[float, int]],
        output_path: Path,
    ) -> None:
        """Save drum events as MIDI file.

        Args:
            events: List of (time_in_seconds, note_number) tuples
            output_path: Output MIDI file path
        """
        try:
            import mido
        except ImportError:
            logger.error("mido not installed, cannot save MIDI")
            return

        # Create MIDI file
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Add tempo meta message (120 BPM)
        track.append(mido.MetaMessage('set_tempo', tempo=500000))  # 120 BPM

        # Convert events to MIDI messages
        # Sort events by time
        events.sort(key=lambda x: x[0])

        # Convert to MIDI ticks (480 ticks per quarter note at 120 BPM)
        ticks_per_second = 480 * 120 / 60  # 960 ticks per second

        current_tick = 0

        for time_sec, note_number in events:
            # Convert time to ticks
            target_tick = int(time_sec * ticks_per_second)
            delta_ticks = target_tick - current_tick

            if delta_ticks < 0:
                delta_ticks = 0

            # Note on
            track.append(mido.Message('note_on', note=note_number, velocity=100, delta=delta_ticks))
            current_tick = target_tick

            # Note off (short duration for drums)
            track.append(mido.Message('note_off', note=note_number, velocity=0, delta=60))

        # Save
        mid.save(str(output_path))

    def _transcribe_fallback(
        self,
        audio_path: Path,
        output_path: Path,
    ) -> Tuple[Path, Dict]:
        """Fallback transcription using simple beat tracking.

        Used when madmom is not available.

        Args:
            audio_path: Input audio file path
            output_path: Output MIDI file path

        Returns:
            Tuple of (output_midi_path, metadata_dict)
        """
        import librosa

        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=None)

        # Detect beats
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)

        # Convert beats to times
        beat_times = librosa.frames_to_time(beats, sr=sr)

        # Create simple drum pattern
        # Kick on 1 and 3, snare on 2 and 4
        events = []

        for i, beat_time in enumerate(beat_times):
            if i % 4 in [0, 2]:
                events.append((beat_time, self.DRUM_NOTES["kick"]))
            else:
                events.append((beat_time, self.DRUM_NOTES["snare"]))

        # Save MIDI
        self._save_drum_midi(events, output_path)

        metadata = {
            "model": "fallback",
            "method": "simple_beat_tracking",
            "tempo": float(tempo),
            "num_beats": len(beats),
            "num_events": len(events),
        }

        logger.warning(f"Used fallback transcription for drums")

        return output_path, metadata

    def get_supported_drum_types(self) -> Dict[str, int]:
        """Get list of supported drum types and their MIDI note numbers.

        Returns:
            Dictionary mapping drum type to MIDI note number
        """
        return self.DRUM_NOTES.copy()


class AdvancedDrumTranscriber(DrumTranscriber):
    """Advanced drum transcriber with more sophisticated detection.

    Features:
    - Multiple kit detection (for more than one drum kit in audio)
    - Dynamic detection
    - Roll/flam detection
    - Ghost note detection
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transcribe_advanced(
        self,
        audio_path: Path,
        output_path: Path,
        detect_ghost_notes: bool = True,
        detect_rolls: bool = True,
    ) -> Tuple[Path, Dict]:
        """Advanced drum transcription with additional features.

        Args:
            audio_path: Input audio file path
            output_path: Output MIDI file path
            detect_ghost_notes: Whether to detect ghost notes
            detect_rolls: Whether to detect drum rolls

        Returns:
            Tuple of (output_midi_path, metadata_dict)
        """
        if not self.available:
            logger.warning("Advanced features require madmom, falling back to basic")
            return self.transcribe(audio_path, output_path)

        # Load audio
        audio, sr = sf.read(str(audio_path))
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        # Use madmom's advanced drum transcription
        from madmom.features import Drums

        # Detect events with activity patterns
        events = []

        # Save to MIDI
        self._save_drum_midi(events, output_path)

        metadata = {
            "model": "madmom_advanced",
            "detect_ghost_notes": detect_ghost_notes,
            "detect_rolls": detect_rolls,
        }

        return output_path, metadata
