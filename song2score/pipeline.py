# Copyright (c) 2026 winjayran
# SPDX-License-Identifier: MIT
"""
Main processing pipeline for song2score.

This module orchestrates the entire workflow:
1. Audio preprocessing
2. Stem separation
3. MIDI transcription
4. MusicXML export
5. PDF rendering (optional)
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from song2score.types import (
    PartType,
    StemConfig,
    TranscriptionConfig,
    ExportConfig,
    ProcessingReport,
)
from song2score.audio.preprocess import AudioPreprocessor
from song2score.separation.demucs import DemucsSeparator
from song2score.separation.strings import StringsSeparator
from song2score.export.musicxml import MusicXMLExporter
from song2score.render.musescore import MuseScoreRenderer

logger = logging.getLogger(__name__)

# Lazy imports for transcribers to avoid TensorFlow warnings on import
def _get_transcribers():
    """Lazy import transcribers to avoid TensorFlow warnings."""
    from song2score.transcription.basic_pitch import (
        BasicPitchTranscriber,
        GuitarTranscriber,
        PianoTranscriber,
        ViolinTranscriber,
    )
    from song2score.transcription.drums import DrumTranscriber
    return (
        BasicPitchTranscriber,
        GuitarTranscriber,
        PianoTranscriber,
        ViolinTranscriber,
        DrumTranscriber,
    )


class Pipeline:
    """Main processing pipeline for song2score."""

    def __init__(
        self,
        output_dir: Path,
        stem_config: Optional[StemConfig] = None,
        transcription_config: Optional[TranscriptionConfig] = None,
        export_config: Optional[ExportConfig] = None,
        device: Optional[str] = None,
    ):
        """Initialize the pipeline.

        Args:
            output_dir: Output directory for all results
            stem_config: Stem separation configuration
            transcription_config: MIDI transcription configuration
            export_config: MusicXML export configuration
            device: Device to use (cpu, cuda, mps)
        """
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.stem_config = stem_config or StemConfig()
        self.transcription_config = transcription_config or TranscriptionConfig()
        self.export_config = export_config or ExportConfig()

        self.device = device

        # Initialize components
        self.preprocessor = AudioPreprocessor()
        self.separator = DemucsSeparator(
            model=self.stem_config.model,
            device=self.device,
        )
        self.strings_separator = StringsSeparator()

        # Lazy initialize transcribers (will be created on first use)
        self._basic_pitch = None
        self._guitar_transcriber = None
        self._piano_transcriber = None
        self._violin_transcriber = None
        self._drum_transcriber = None

        self.exporter = MusicXMLExporter(self.export_config)
        self.renderer = MuseScoreRenderer()

        # Report
        self.report = ProcessingReport(
            output_dir=self.output_dir,
        )

    @property
    def basic_pitch(self):
        """Lazy load basic pitch transcriber."""
        if self._basic_pitch is None:
            BasicPitchTranscriber, *_ = _get_transcribers()
            self._basic_pitch = BasicPitchTranscriber(
                confidence_threshold=self.transcription_config.confidence_threshold,
                minimum_note_length=self.transcription_config.minimum_note_length,
                midi_tempo=self.transcription_config.midi_tempo,
            )
        return self._basic_pitch

    @property
    def guitar_transcriber(self):
        """Lazy load guitar transcriber."""
        if self._guitar_transcriber is None:
            _, GuitarTranscriber, *_ = _get_transcribers()
            self._guitar_transcriber = GuitarTranscriber(
                confidence_threshold=self.transcription_config.confidence_threshold,
            )
        return self._guitar_transcriber

    @property
    def piano_transcriber(self):
        """Lazy load piano transcriber."""
        if self._piano_transcriber is None:
            _, _, PianoTranscriber, *_ = _get_transcribers()
            self._piano_transcriber = PianoTranscriber(
                confidence_threshold=self.transcription_config.confidence_threshold,
            )
        return self._piano_transcriber

    @property
    def violin_transcriber(self):
        """Lazy load violin transcriber."""
        if self._violin_transcriber is None:
            _, _, _, ViolinTranscriber, _ = _get_transcribers()
            self._violin_transcriber = ViolinTranscriber(
                confidence_threshold=self.transcription_config.confidence_threshold,
            )
        return self._violin_transcriber

    @property
    def drum_transcriber(self):
        """Lazy load drum transcriber."""
        if self._drum_transcriber is None:
            *_, DrumTranscriber = _get_transcribers()
            self._drum_transcriber = DrumTranscriber()
        return self._drum_transcriber

    def run(
        self,
        input_path: Path,
        parts: Optional[List[PartType]] = None,
    ) -> ProcessingReport:
        """Run the full pipeline.

        Args:
            input_path: Input audio file path
            parts: List of parts to process (None = all available)

        Returns:
            ProcessingReport with results and metadata
        """
        start_time = time.time()

        # Ensure input_path is absolute
        input_path = Path(input_path).resolve()
        self.report.input_file = input_path

        logger.info(f"Starting pipeline for {input_path}")

        try:
            # Step 1: Preprocess audio
            logger.info("Step 1: Preprocessing audio...")
            preprocessed_path = self.preprocessor.preprocess(
                input_path,
                self.output_dir / "preprocessed.wav",
            )

            # Step 2: Separate stems
            logger.info("Step 2: Separating stems...")
            stems = self.separator.separate(
                preprocessed_path,
                self.output_dir / "stems",
                parts=parts,
            )

            self.report.stems_produced = stems

            # Step 3: Process strings from "other" if needed
            if PartType.OTHER in stems and (parts is None or PartType.STRINGS in parts):
                logger.info("Step 2b: Extracting strings from 'other' stem...")
                strings_stems = self.strings_separator.analyze_and_separate(
                    stems[PartType.OTHER],
                    self.output_dir / "stems",
                )

                if strings_stems:
                    stems.update(strings_stems)
                    self.report.stems_produced.update(strings_stems)

            # Step 4: Transcribe each stem to MIDI
            logger.info("Step 3: Transcribing stems to MIDI...")
            midi_dir = self.output_dir / "midi"
            midi_dir.mkdir(exist_ok=True)

            for part_type, stem_path in stems.items():
                if parts and part_type not in parts:
                    continue

                midi_path = midi_dir / f"{part_type.value}.mid"

                try:
                    if part_type == PartType.DRUMS:
                        # Use specialized drum transcriber
                        _, metadata = self.drum_transcriber.transcribe(stem_path, midi_path)

                    elif part_type == PartType.GUITAR:
                        # Use specialized guitar transcriber
                        _, metadata = self.guitar_transcriber.transcribe_guitar(stem_path, midi_path)

                    elif part_type == PartType.PIANO:
                        # Use specialized piano transcriber
                        _, metadata = self.piano_transcriber.transcribe_piano(stem_path, midi_path)

                    elif part_type == PartType.STRINGS:
                        # Use specialized violin/strings transcriber
                        _, metadata = self.violin_transcriber.transcribe_violin(stem_path, midi_path)

                    else:
                        # Use basic pitch for vocals, bass, other
                        _, metadata = self.basic_pitch.transcribe(
                            stem_path,
                            midi_path,
                            part_type,
                        )

                    self.report.midi_produced[part_type] = midi_path

                except Exception as e:
                    logger.error(f"Failed to transcribe {part_type}: {e}")
                    self.report.errors.append(f"Transcription failed for {part_type}: {e}")

            # Step 5: Export to MusicXML if we have MIDI files
            if self.report.midi_produced and self.export_config.parts:
                logger.info("Step 4: Exporting to MusicXML...")
                musicxml_dir = self.output_dir / "musicxml"
                musicxml_dir.mkdir(exist_ok=True)

                musicxml_path = musicxml_dir / "score.musicxml"

                try:
                    output_path, metadata = self.exporter.export(
                        self.report.midi_produced,
                        musicxml_path,
                    )

                    self.report.musicxml_produced = output_path

                except Exception as e:
                    logger.error(f"Failed to export MusicXML: {e}")
                    self.report.errors.append(f"MusicXML export failed: {e}")

            # Step 6: Render PDF if MuseScore is available
            if self.report.musicxml_produced and self.renderer.is_available():
                logger.info("Step 5: Rendering to PDF...")
                render_dir = self.output_dir / "render"
                render_dir.mkdir(exist_ok=True)

                pdf_path = render_dir / "score.pdf"

                try:
                    output_path, _ = self.renderer.render_to_pdf(
                        self.report.musicxml_produced,
                        pdf_path,
                    )

                    logger.info(f"PDF rendered to {output_path}")

                except Exception as e:
                    logger.error(f"Failed to render PDF: {e}")
                    self.report.warnings.append(f"PDF rendering failed: {e}")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.report.errors.append(f"Pipeline failed: {e}")

        # Calculate processing time
        self.report.processing_time_seconds = time.time() - start_time

        # Save report
        self._save_report()

        logger.info(f"Pipeline completed in {self.report.processing_time_seconds:.2f}s")

        return self.report

    def _save_report(self) -> None:
        """Save the processing report to JSON."""
        import json

        report_path = self.output_dir / "report.json"

        # Convert to dict for JSON serialization
        report_dict = {
            "input_file": str(self.report.input_file),
            "output_dir": str(self.report.output_dir),
            "stems_produced": {
                k.value: str(v) for k, v in self.report.stems_produced.items()
            },
            "midi_produced": {
                k.value: str(v) for k, v in self.report.midi_produced.items()
            },
            "musicxml_produced": str(self.report.musicxml_produced) if self.report.musicxml_produced else None,
            "errors": self.report.errors,
            "warnings": self.report.warnings,
            "processing_time_seconds": self.report.processing_time_seconds,
        }

        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"Report saved to {report_path}")


class QuickTranscribe:
    """Quick transcription shortcut for common use cases."""

    @staticmethod
    def transcribe(
        input_path: Path,
        output_dir: Path,
        parts: Optional[List[PartType]] = None,
        device: str = "cpu",
    ) -> ProcessingReport:
        """Quick transcribe a song to MIDI.

        Args:
            input_path: Input audio file path
            output_dir: Output directory
            parts: Parts to transcribe (None = all)
            device: Device to use

        Returns:
            ProcessingReport
        """
        pipeline = Pipeline(
            output_dir=output_dir,
            device=device,
        )

        return pipeline.run(input_path, parts)

    @staticmethod
    def to_score(
        input_path: Path,
        output_dir: Path,
        parts: Optional[List[PartType]] = None,
        instrument_map: Optional[Dict[PartType, str]] = None,
        device: str = "cpu",
    ) -> ProcessingReport:
        """Transcribe and export to MusicXML score.

        Args:
            input_path: Input audio file path
            output_dir: Output directory
            parts: Parts to include in score
            instrument_map: Instrument mapping for re-orchestration
            device: Device to use

        Returns:
            ProcessingReport
        """
        export_config = ExportConfig(
            parts=parts or [PartType.VOCALS, PartType.GUITAR, PartType.PIANO, PartType.BASS],
            instrument_map=instrument_map or {},
        )

        pipeline = Pipeline(
            output_dir=output_dir,
            export_config=export_config,
            device=device,
        )

        return pipeline.run(input_path, parts)
