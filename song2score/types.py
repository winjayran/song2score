# Copyright (c) 2026 winjayran
# SPDX-License-Identifier: MIT
"""Common types and enums for song2score."""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PartType(str, Enum):
    """Supported part types for separation and transcription."""

    VOCALS = "vocals"
    DRUMS = "drums"
    BASS = "bass"
    GUITAR = "guitar"
    PIANO = "piano"
    STRINGS = "strings"
    OTHER = "other"


class StemConfig(BaseModel):
    """Configuration for stem separation."""

    num_stems: int = Field(default=4, ge=4, le=6, description="Number of stems (4 or 6)")
    parts: List[PartType] = Field(
        default_factory=lambda: [PartType.VOCALS, PartType.DRUMS, PartType.BASS, PartType.OTHER],
        description="Parts to extract",
    )
    model: str = Field(default="htdemucs", description="Demucs model name")


class TranscriptionConfig(BaseModel):
    """Configuration for MIDI transcription."""

    model: str = Field(default="basic-pitch", description="Transcription model")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    minimum_note_length: float = Field(default=0.05, description="Minimum note duration in seconds")
    midi_tempo: int = Field(default=120, ge=20, le=300, description="Default MIDI tempo")


class ExportConfig(BaseModel):
    """Configuration for MusicXML export."""

    title: str = Field(default="Transcribed Score")
    composer: str = Field(default="")
    parts: List[PartType] = Field(default_factory=list)
    instrument_map: Dict[PartType, str] = Field(default_factory=dict)
    guitar_tab: bool = Field(default=False)
    quantization: int = Field(default=16, description="Quantization grid (4 = quarter, 8 = eighth, etc.)")


class DeviceConfig(BaseModel):
    """Configuration for device selection."""

    device: str = Field(default="cpu", description="Device: cpu, cuda, mps")
    batch_size: int = Field(default=1, ge=1, description="Batch size for processing")
    workers: int = Field(default=1, ge=1, description="Number of worker threads")


class ProcessingReport(BaseModel):
    """Report of processing results."""

    input_file: Optional[Path] = None
    output_dir: Path
    stems_produced: Dict[PartType, Path] = Field(default_factory=dict)
    midi_produced: Dict[PartType, Path] = Field(default_factory=dict)
    musicxml_produced: Optional[Path] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    processing_time_seconds: float = 0.0

    class Config:
        arbitrary_types_allowed = True


# Standard MIDI program numbers for common instruments
MIDI_INSTRUMENTS = {
    # Piano
    "acoustic_grand_piano": 0,
    "bright_acoustic_piano": 1,
    "electric_grand_piano": 2,
    "electric_piano_1": 4,
    # Guitar
    "acoustic_guitar_nylon": 24,
    "acoustic_guitar_steel": 25,
    "electric_guitar_jazz": 26,
    "electric_guitar_clean": 27,
    "electric_guitar_muted": 28,
    "electric_guitar_overdriven": 29,
    "electric_guitar_distortion": 30,
    # Bass
    "electric_bass_finger": 33,
    "electric_bass_pick": 34,
    "fretless_bass": 35,
    "slap_bass_1": 36,
    "slap_bass_2": 37,
    "synth_bass_1": 38,
    "synth_bass_2": 39,
    # Strings
    "violin": 40,
    "viola": 41,
    "cello": 42,
    "contrabass": 43,
    "tremolo_strings": 44,
    "pizzicato_strings": 45,
    "orchestral_harp": 46,
    "timpani": 47,
    "string_ensemble_1": 48,
    "string_ensemble_2": 49,
    # Brass
    "trumpet": 56,
    "trombone": 57,
    "tuba": 58,
    "muted_trumpet": 59,
    "french_horn": 60,
    "brass_section": 61,
    "synth_brass_1": 62,
    "synth_brass_2": 63,
    # Winds
    "soprano_sax": 64,
    "alto_sax": 65,
    "tenor_sax": 66,
    "baritone_sax": 67,
    "flute": 73,
    "clarinet": 71,
    "ocarina": 79,
    # Drums
    "standard_kit": 0,  # Channel 10
    # Vocals (as synthesized)
    "choir_aahs": 80,
    "voice_oohs": 81,
    "synth_choir": 82,
}
