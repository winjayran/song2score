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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

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
from song2score.separation.classifier import InstrumentClassifier, classify_stem
from song2score.separation.refinement import StemRefiner
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
        parallel_transcription: bool = False,  # Disabled by default due to TensorFlow/Basic Pitch thread safety
        max_transcription_workers: int = 3,
        stem_remap: Optional[Dict[PartType, PartType]] = None,
        refine_stems: bool = False,
    ):
        """Initialize the pipeline.

        Args:
            output_dir: Output directory for all results
            stem_config: Stem separation configuration
            transcription_config: MIDI transcription configuration
            export_config: MusicXML export configuration
            device: Device to use (cpu, cuda, mps)
            parallel_transcription: Whether to transcribe stems in parallel
            max_transcription_workers: Max parallel transcription workers
            stem_remap: Manual mapping of stem files to part types (e.g., {PartType.VOCALS: PartType.DRUMS})
            refine_stems: Whether to apply stem refinement to clean up mixed audio
        """
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.stem_config = stem_config or StemConfig()
        self.transcription_config = transcription_config or TranscriptionConfig()
        self.export_config = export_config or ExportConfig()

        self.device = device
        self.parallel_transcription = parallel_transcription
        self.max_transcription_workers = max_transcription_workers
        self.stem_remap = stem_remap or {}
        self.refine_stems = refine_stems

        # Initialize components
        self.preprocessor = AudioPreprocessor()
        self.separator = DemucsSeparator(
            model=self.stem_config.model,
            device=self.device,
        )
        self.strings_separator = StringsSeparator()
        self.stem_refiner = StemRefiner() if refine_stems else None

        # Lazy initialize transcribers (will be created on first use)
        self._basic_pitch = None
        self._guitar_transcriber = None
        self._piano_transcriber = None
        self._violin_transcriber = None
        self._drum_transcriber = None

        self.exporter = MusicXMLExporter(self.export_config)
        self.renderer = MuseScoreRenderer()

        # Instrument classifier for improved detection
        self.classifier = InstrumentClassifier()

        # Report
        self.report = ProcessingReport(
            output_dir=self.output_dir,
        )

    def _verify_and_correct_stem_classification(
        self,
        stems: Dict[PartType, Path],
        auto_correct: bool = True,
        min_confidence: float = 0.65,
    ) -> Dict[PartType, Path]:
        """Verify and optionally correct stem classifications using audio analysis.

        This analyzes the actual content of each stem and can reassign stems
        to different part types based on their actual instrument content.
        This helps when Demucs misassigns instruments (e.g., guitar in "other").

        NOTE: This is CONSERVATIVE - it only reclassifies the "other" stem to avoid
        breaking valid stem assignments. For more aggressive reclassification, use
        the --remap-stems option.

        Args:
            stems: Dictionary of PartType to stem paths
            auto_correct: If True, reclassify "other" stem when confidence is high
            min_confidence: Minimum confidence for auto-correction

        Returns:
            Corrected stems dictionary
        """
        # Only perform reclassification on "other" stem to find missing instruments
        # This is conservative - we only reclassify "other" to avoid breaking valid stems
        if PartType.OTHER in stems:
            other_stem_path = stems[PartType.OTHER]
            detected_part, confidence = classify_stem(other_stem_path)
            logger.info(f"Stem 'other' content detected as '{detected_part.value}' (confidence: {confidence:.2f})")

            # Only reclassify if confidence is high AND detected type doesn't already exist
            if auto_correct and confidence >= min_confidence and detected_part != PartType.OTHER:
                if detected_part not in stems:
                    logger.info(f"Reclassifying 'other' -> '{detected_part.value}' (confidence: {confidence:.2f})")
                    # Create new stems dict with reclassification
                    corrected_stems = dict(stems)
                    del corrected_stems[PartType.OTHER]
                    corrected_stems[detected_part] = other_stem_path
                    return corrected_stems
                else:
                    logger.info(f"Not reclassifying 'other' -> '{detected_part.value}' (type already exists)")
            else:
                logger.info(f"Keeping 'other' stem (confidence: {confidence:.2f} < {min_confidence} or no correction)")

        return stems

    def _apply_stem_remapping(
        self,
        stems: Dict[PartType, Path],
        remap: Dict[PartType, PartType],
    ) -> Dict[PartType, Path]:
        """Apply manual stem remapping based on user specification.

        This allows users to manually correct Demucs misclassifications by specifying
        which stem file should be treated as which instrument type.

        Args:
            stems: Dictionary of PartType to stem paths
            remap: Dictionary mapping original PartType to target PartType

        Returns:
            Remapped stems dictionary

        Example:
            If remap = {PartType.VOCALS: PartType.DRUMS, PartType.OTHER: PartType.VOCALS}
            Then vocals.wav will be transcribed as drums, and other.wav as vocals.
        """
        remapped_stems = {}
        remapping_log = []

        # Track which target types are already assigned
        assigned_targets = set()

        for original_type, stem_path in stems.items():
            # Check if this stem should be remapped
            if original_type in remap:
                target_type = remap[original_type]
                remapping_log.append(f"{original_type.value} -> {target_type.value}")

                # Check if target is already assigned (conflict)
                if target_type in assigned_targets:
                    logger.warning(f"Conflict: Multiple stems mapped to {target_type.value}, using last one")

                remapped_stems[target_type] = stem_path
                assigned_targets.add(target_type)
            else:
                # Check if this original type was the target of a remapping
                # If someone else mapped to this type, we need to skip it
                is_target_of_remap = any(remap[t] == original_type for t in remap if t in stems)
                if is_target_of_remap:
                    logger.info(f"Skipping {original_type.value} stem (remapped to another type)")
                else:
                    remapped_stems[original_type] = stem_path
                    assigned_targets.add(original_type)

        # Log remapping
        if remapping_log:
            logger.info(f"Applied manual stem remapping:")
            for log_entry in remapping_log:
                logger.info(f"  {log_entry}")

        return remapped_stems

    def _refine_stems(
        self,
        stems: Dict[PartType, Path],
    ) -> Dict[PartType, Path]:
        """Refine stems to remove unwanted content from other instruments.

        This applies HPSS and frequency filtering to clean up stems that
        contain residual audio from other instruments.

        Args:
            stems: Dictionary of PartType to stem paths

        Returns:
            Dictionary of PartType to refined stem paths
        """
        if not self.stem_refiner:
            return stems

        refined_stems = {}

        for part_type, stem_path in stems.items():
            try:
                # Apply refinement
                refined_path, metadata = self.stem_refiner.refine_stem(
                    stem_path,
                    part_type,
                    stem_path,  # Overwrite original
                )
                refined_stems[part_type] = refined_path

                logger.info(f"Refined {part_type.value} stem")
                if metadata.get("frequency_filter"):
                    logger.info(f"  Frequency filter: {metadata['frequency_filter']}")

            except Exception as e:
                logger.warning(f"Failed to refine {part_type.value} stem: {e}")
                # Keep original stem
                refined_stems[part_type] = stem_path

        return refined_stems

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

    def _transcribe_stem(
        self,
        part_type: PartType,
        stem_path: Path,
        midi_path: Path,
    ) -> Tuple[PartType, Optional[Path], Optional[Dict], Optional[str]]:
        """Transcribe a single stem to MIDI.

        Args:
            part_type: Type of part to transcribe
            stem_path: Path to stem audio file
            midi_path: Output MIDI file path

        Returns:
            Tuple of (part_type, midi_path, metadata, error)
        """
        try:
            if part_type == PartType.DRUMS:
                _, metadata = self.drum_transcriber.transcribe(stem_path, midi_path)

            elif part_type == PartType.GUITAR:
                _, metadata = self.guitar_transcriber.transcribe_guitar(stem_path, midi_path)

            elif part_type == PartType.PIANO:
                _, metadata = self.piano_transcriber.transcribe_piano(stem_path, midi_path)

            elif part_type == PartType.STRINGS:
                _, metadata = self.violin_transcriber.transcribe_violin(stem_path, midi_path)

            else:
                # Use basic pitch for vocals, bass, other
                _, metadata = self.basic_pitch.transcribe(
                    stem_path,
                    midi_path,
                    part_type,
                )

            return (part_type, midi_path, metadata, None)

        except ValueError as e:
            # ValueError is raised for empty/silent audio - skip this stem
            logger.warning(f"Skipping {part_type} transcription: {e}")
            return (part_type, None, None, f"Skipped: {e}")
        except Exception as e:
            logger.error(f"Failed to transcribe {part_type}: {e}")
            return (part_type, None, None, str(e))

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

            # Step 2b: Apply manual stem remapping (if specified)
            if self.stem_remap:
                logger.info("Step 2b: Applying manual stem remapping...")
                stems = self._apply_stem_remapping(stems, self.stem_remap)
                self.report.stems_produced = stems

            # Step 2c: Verify and correct stem classifications based on audio content
            logger.info("Step 2c: Verifying stem classifications...")
            stems = self._verify_and_correct_stem_classification(
                stems,
                auto_correct=True,  # Enable auto-correction
                min_confidence=0.65,  # Minimum confidence for correction
            )
            self.report.stems_produced = stems

            # Step 2d: Refine stems to remove unwanted content (if enabled)
            if self.refine_stems:
                logger.info("Step 2d: Refining stems to remove unwanted content...")
                stems = self._refine_stems(stems)
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

            # Filter stems to process
            stems_to_process = [
                (part_type, stem_path)
                for part_type, stem_path in stems.items()
                if not parts or part_type in parts
            ]

            if self.parallel_transcription and len(stems_to_process) > 1:
                # Parallel transcription
                logger.info(f"Transcribing {len(stems_to_process)} stems in parallel...")
                with ThreadPoolExecutor(max_workers=self.max_transcription_workers) as executor:
                    futures = {
                        executor.submit(
                            self._transcribe_stem,
                            part_type,
                            stem_path,
                            midi_dir / f"{part_type.value}.mid",
                        ): part_type
                        for part_type, stem_path in stems_to_process
                    }

                    for future in as_completed(futures):
                        part_type, midi_path, metadata, error = future.result()
                        if midi_path:
                            self.report.midi_produced[part_type] = midi_path
                        if error:
                            self.report.errors.append(f"Transcription failed for {part_type}: {error}")
            else:
                # Sequential transcription
                for part_type, stem_path in stems_to_process:
                    midi_path = midi_dir / f"{part_type.value}.mid"
                    part_type, midi_path, metadata, error = self._transcribe_stem(
                        part_type, stem_path, midi_path
                    )
                    if midi_path:
                        self.report.midi_produced[part_type] = midi_path
                    if error:
                        self.report.errors.append(f"Transcription failed for {part_type}: {error}")

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
