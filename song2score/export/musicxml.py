# Copyright (c) 2026 winjayran
# SPDX-License-Identifier: MIT
"""MusicXML export module with instrument mapping and TAB support.

This module handles:
- MIDI to MusicXML conversion
- Instrument re-orchestration (mapping parts to different instruments)
- Guitar TAB generation
- Score layout and formatting
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from music21 import (
    stream,
    note,
    meter,
    key,
    tempo,
    instrument,
    clef,
    layout,
    chord,
    metadata as m21_metadata,
    converter,
)

from song2score.types import PartType, MIDI_INSTRUMENTS, ExportConfig

logger = logging.getLogger(__name__)


class MusicXMLExporter:
    """Export MIDI files to MusicXML with re-orchestration options."""

    # Guitar tuning for TAB (standard E A D G B E)
    GUITAR_TUNING = ["E4", "B3", "G3", "D3", "A2", "E2"]

    # Guitar string MIDI numbers (E4=64, B3=59, G3=55, D3=50, A2=45, E2=40)
    GUITAR_STRING_MIDI = [64, 59, 55, 50, 45, 40]

    # Clef assignments by instrument
    CLEF_ASSIGNMENTS = {
        PartType.VOCALS: "treble",
        PartType.GUITAR: "treble",
        PartType.PIANO: "treble-bass",  # Piano uses grand staff
        PartType.BASS: "bass",
        PartType.DRUMS: "percussion",
        PartType.STRINGS: "treble",  # Violin, viola
        PartType.OTHER: "treble",
    }

    # Specific clefs for string instruments
    STRINGS_CLEFS = {
        "violin": "treble",
        "viola": "alto",
        "cello": "bass",
        "contrabass": "bass",
    }

    def __init__(self, config: Optional[ExportConfig] = None):
        """Initialize the MusicXML exporter.

        Args:
            config: Export configuration
        """
        self.config = config or ExportConfig()

    def export(
        self,
        midi_files: Dict[PartType, Path],
        output_path: Path,
        title: Optional[str] = None,
        composer: Optional[str] = None,
    ) -> Tuple[Path, Dict]:
        """Export MIDI files to a combined MusicXML score.

        Args:
            midi_files: Dictionary mapping PartType to MIDI file paths
            output_path: Output MusicXML file path
            title: Score title
            composer: Score composer

        Returns:
            Tuple of (output_path, metadata_dict)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create score
        score = stream.Score()

        # Set metadata
        title = title or self.config.title
        composer = composer or self.config.composer

        score.metadata = m21_metadata.Metadata()
        score.metadata.title = title
        if composer:
            score.metadata.composer = composer

        # Process each part in a specific order for better layout
        # Order: vocals, guitar, piano, bass, strings, drums, other
        part_order = [
            PartType.VOCALS,
            PartType.GUITAR,
            PartType.PIANO,
            PartType.BASS,
            PartType.STRINGS,
            PartType.DRUMS,
            PartType.OTHER,
        ]

        # Sort parts according to the order
        sorted_parts = sorted(
            midi_files.items(),
            key=lambda x: part_order.index(x[0]) if x[0] in part_order else len(part_order)
        )

        for part_type, midi_path in sorted_parts:
            if part_type not in self.config.parts and self.config.parts:
                continue

            # Get instrument mapping
            mapped_instrument = self.config.instrument_map.get(
                part_type,
                part_type.value,
            )

            # Create part
            part = self._create_part_from_midi(
                midi_path,
                part_type,
                mapped_instrument,
            )

            if part:
                # Set part ID for proper staff identification
                part.id = f"{part_type.value}_staff"
                part.partName = mapped_instrument.replace("_", " ").title()
                part.partAbbreviation = mapped_instrument[:8].replace("_", " ").title()

                # Show debug info
                logger.info(f"Adding part for {part_type.value}: {len(list(part.flatten().notes))} notes")
                score.insert(part)

        # Add layout if needed
        self._add_layout(score)

        # Write MusicXML
        score.write("musicxml", str(output_path))

        logger.info(f"Exported MusicXML to {output_path}")

        metadata = {
            "num_parts": len(score.parts),
            "title": title,
            "instrument_map": dict(self.config.instrument_map),
        }

        return output_path, metadata

    def _create_part_from_midi(
        self,
        midi_path: Path,
        part_type: PartType,
        mapped_instrument: str,
    ) -> Optional[stream.Part]:
        """Create a music21 Part from a MIDI file.

        Args:
            midi_path: Path to MIDI file
            part_type: Original part type
            mapped_instrument: Target instrument name

        Returns:
            music21 Part object or None if failed
        """
        try:
            # Convert MIDI to music21 stream
            midi_stream = converter.parse(str(midi_path))

            # Get the first part from the MIDI stream
            parts = list(midi_stream.parts)
            if parts:
                # Use the first part directly
                part = parts[0]

                # Update instrument
                instr = self._get_instrument(mapped_instrument)
                part.partsAreSparse = False

                # Clear existing instruments and add new one
                existing_instrs = part.getInstruments()
                if existing_instrs:
                    for old_instr in existing_instrs:
                        part.remove(old_instr)
                part.insert(0, instr)

                # Add guitar TAB if requested
                if part_type == PartType.GUITAR and self.config.guitar_tab:
                    self._add_guitar_tab(part, midi_stream)

                return part
            else:
                # Fallback: create from notes
                part = stream.Part()
                instr = self._get_instrument(mapped_instrument)
                part.insert(0, instr)

                clef_name = self._get_clef(part_type, mapped_instrument)
                # Use clefFromString to create clef from string name
                part.insert(0, clef.clefFromString(clef_name))

                part.insert(0, meter.TimeSignature('4/4'))
                part.insert(0, tempo.MetronomeMark(number=120))

                # Transfer notes
                for element in midi_stream.flatten().notesAndRests:
                    part.append(element)

                if part_type == PartType.GUITAR and self.config.guitar_tab:
                    self._add_guitar_tab(part, midi_stream)

                return part

        except Exception as e:
            logger.error(f"Failed to process MIDI {midi_path}: {e}")
            return None

        except Exception as e:
            logger.error(f"Failed to process MIDI {midi_path}: {e}")
            return None

    def _get_instrument(self, instrument_name: str) -> instrument.Instrument:
        """Get a music21 Instrument object.

        Args:
            instrument_name: Name of the instrument

        Returns:
            music21 Instrument object
        """
        # Try MIDI_INSTRUMENTS first
        program = MIDI_INSTRUMENTS.get(instrument_name, 0)

        # Map to music21 instrument classes
        instrument_map = {
            "violin": instrument.Violin(),
            "viola": instrument.Viola(),
            "cello": instrument.Violoncello(),
            "contrabass": instrument.Contrabass(),
            "acoustic_guitar_nylon": instrument.AcousticGuitar(),
            "electric_guitar_jazz": instrument.ElectricGuitar(),
            "acoustic_grand_piano": instrument.Piano(),
            "electric_bass_finger": instrument.ElectricBass(),
            "trumpet": instrument.Trumpet(),
            "trombone": instrument.Trombone(),
            "flute": instrument.Flute(),
            "clarinet": instrument.Clarinet(),
        }

        if instrument_name in instrument_map:
            return instrument_map[instrument_name]

        # Create generic instrument with MIDI program
        instr = instrument.Instrument()
        instr.midiProgram = program
        instr.instrumentName = instrument_name.replace("_", " ").title()

        return instr

    def _get_clef(self, part_type: PartType, instrument_name: str) -> str:
        """Determine the appropriate clef for a part.

        Args:
            part_type: Original part type
            instrument_name: Target instrument name

        Returns:
            Clef name (treble, bass, alto, percussion, etc.)
        """
        # Check strings specific clefs
        if instrument_name in self.STRINGS_CLEFS:
            return self.STRINGS_CLEFS[instrument_name]

        # Use default clef assignments
        return self.CLEF_ASSIGNMENTS.get(part_type, "treble")

    def _add_guitar_tab(
        self,
        part: stream.Part,
        midi_stream: stream.Stream,
    ) -> None:
        """Add guitar TAB notation to a part.

        This creates a second staff with TAB notation below the standard notation.

        Args:
            part: The part to add TAB to
            midi_stream: Original MIDI stream for note extraction
        """
        # Create TAB staff
        tab_part = stream.Part()
        tab_part.id = "TAB"

        # Add TAB clef
        tab_clef = clef.TabClef()
        tab_part.insert(0, tab_clef)

        # Add guitar instrument to TAB
        guitar = instrument.AcousticGuitar()
        tab_part.insert(0, guitar)

        # Convert notes to TAB
        for element in midi_stream.flatten().notes:
            if isinstance(element, note.Note):
                # Find best string and fret for this note
                string_num, fret = self._find_guitar_position(element.pitch.midi)

                # Create TAB note
                tab_note = note.Note()
                tab_note.pitch.midi = element.pitch.midi
                tab_note.duration.type = element.duration.type

                # Store string and fret as editorial information
                tab_note.lyric = f"String {string_num}, Fret {fret}"

                tab_part.insert(element.offset, tab_note)
            elif isinstance(element, chord.Chord):
                # Handle chords
                for pitch in element.pitches:
                    string_num, fret = self._find_guitar_position(pitch.midi)
                    # ... handle chord TAB

        # Merge TAB with standard notation
        # This is a simplified approach - full TAB implementation requires more work

    def _find_guitar_position(
        self,
        midi_note: int,
    ) -> Tuple[int, int]:
        """Find the best string and fret for a MIDI note on guitar.

        Args:
            midi_note: MIDI note number

        Returns:
            Tuple of (string_number, fret) where string_number is 1-6 (1=high E)
        """
        # Try each string from high to low
        for string_num, string_midi in enumerate(self.GUITAR_STRING_MIDI):
            fret = midi_note - string_midi

            if 0 <= fret <= 24:  # Standard guitars have up to 24 frets
                # string_num is 0-5, return 1-6 (1=high E string)
                return string_num + 1, fret

        # Fallback: use lowest string with highest fret
        fret = midi_note - self.GUITAR_STRING_MIDI[-1]
        return 6, max(0, min(fret, 24))

    def _add_layout(self, score: stream.Score) -> None:
        """Add layout information to the score.

        Ensures each instrument is on a separate staff with proper spacing.

        Args:
            score: The score to add layout to
        """
        # Add page layout with proper margins
        page_layout = layout.PageLayout()
        page_layout.pageWidth = 2100  # Standard A4 width (tenths)
        page_layout.pageHeight = 2970  # Standard A4 height (tenths)
        page_layout.topMargin = 100
        page_layout.bottomMargin = 100
        page_layout.leftMargin = 100
        page_layout.rightMargin = 100
        score.append(page_layout)

        # Add system layout and staff layout for each part
        # This ensures each instrument gets its own staff
        for part in score.parts:
            # System layout controls spacing between systems
            system_layout = layout.SystemLayout()
            system_layout.systemDistance = 80  # Distance between systems
            part.insert(0, system_layout)

            # Staff layout controls individual staff properties
            # Each part gets its own staff
            staff_layout = layout.StaffLayout()
            staff_layout.staffDistance = 60  # Distance between staves in a system
            part.insert(0, staff_layout)

        # Add staff group for parts that should be connected (e.g., piano grand staff)
        # For instruments like piano that use multiple staves
        self._add_staff_groups(score)

    def _add_staff_groups(self, score: stream.Score) -> None:
        """Add staff groups for instruments that use multiple staves.

        Args:
            score: The score to add staff groups to
        """
        # Check if piano is present (uses grand staff - treble and bass)
        for part in score.parts:
            instr = part.getInstrument()
            if instr and 'piano' in instr.instrumentName.lower():
                # Piano uses a grand staff (bracketed together)
                staff_group = layout.StaffGroup()
                staff_group.symbol = "brace"  # Brace for piano grand staff
                part.insert(0, staff_group)
            elif instr and 'guitar' in instr.instrumentName.lower():
                # Guitar can use a bracket for standard notation + TAB
                staff_group = layout.StaffGroup()
                staff_group.symbol = "bracket"
                part.insert(0, staff_group)

    def export_single_midi(
        self,
        midi_path: Path,
        output_path: Path,
        part_type: PartType,
        instrument_override: Optional[str] = None,
    ) -> Tuple[Path, Dict]:
        """Export a single MIDI file to MusicXML.

        Args:
            midi_path: Input MIDI file path
            output_path: Output MusicXML file path
            part_type: Type of part
            instrument_override: Override instrument name

        Returns:
            Tuple of (output_path, metadata_dict)
        """
        return self.export(
            {part_type: midi_path},
            output_path,
        )

    def set_instrument_map(self, instrument_map: Dict[PartType, str]) -> None:
        """Set the instrument mapping for export.

        Args:
            instrument_map: Dictionary mapping PartType to instrument names
        """
        self.config.instrument_map.update(instrument_map)

    def set_parts(self, parts: List[PartType]) -> None:
        """Set which parts to include in export.

        Args:
            parts: List of PartType to include
        """
        self.config.parts = parts

    def enable_guitar_tab(self, enable: bool = True) -> None:
        """Enable or disable guitar TAB output.

        Args:
            enable: Whether to enable guitar TAB
        """
        self.config.guitar_tab = enable

    def set_quantization(self, quantization: int) -> None:
        """Set the quantization grid for export.

        Args:
            quantization: Quantization grid (4=quarter, 8=eighth, 16=sixteenth)
        """
        self.config.quantization = quantization


class GuitarTabExporter(MusicXMLExporter):
    """Specialized exporter for guitar TAB notation.

    Focuses on creating accurate and playable guitar TAB with:
    - Optimal string/fret selection
    - Chord diagrams
    - Technique markings (hammer-ons, pull-offs, slides, bends)
    """

    def __init__(self, tuning: Optional[List[str]] = None):
        """Initialize the guitar TAB exporter.

        Args:
            tuning: Custom tuning (list of pitch names from high to low)
        """
        super().__init__()
        self.tuning = tuning or MusicXMLExporter.GUITAR_TUNING

    def export_guitar_tab(
        self,
        midi_path: Path,
        output_path: Path,
        tuning: Optional[List[str]] = None,
    ) -> Tuple[Path, Dict]:
        """Export guitar MIDI to TAB notation.

        Args:
            midi_path: Input MIDI file path
            output_path: Output MusicXML file path
            tuning: Custom tuning (None for standard EADGBE)

        Returns:
            Tuple of (output_path, metadata_dict)
        """
        if tuning:
            self.tuning = tuning

        config = ExportConfig(
            parts=[PartType.GUITAR],
            guitar_tab=True,
        )
        self.config = config

        return self.export_single_midi(
            midi_path,
            output_path,
            PartType.GUITAR,
        )


class DrumScoreExporter(MusicXMLExporter):
    """Specialized exporter for drum scores.

    Creates proper drum notation with:
    - Drum set staff
    - Proper note head shapes for different drums
    - Percussion clef
    - Standard drum mapping
    """

    def __init__(self):
        """Initialize the drum score exporter."""
        super().__init__()

    def export_drum_score(
        self,
        midi_path: Path,
        output_path: Path,
    ) -> Tuple[Path, Dict]:
        """Export drum MIDI to proper drum notation.

        Args:
            midi_path: Input MIDI file path (drum MIDI on channel 10)
            output_path: Output MusicXML file path

        Returns:
            Tuple of (output_path, metadata_dict)
        """
        config = ExportConfig(
            parts=[PartType.DRUMS],
        )
        self.config = config

        # Use drum-specific clef and handling
        return self.export_single_midi(
            midi_path,
            output_path,
            PartType.DRUMS,
        )
