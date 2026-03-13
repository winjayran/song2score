# Copyright (c) 2026 winjayran
# SPDX-License-Identifier: MIT
"""Stem separation using Demucs with memory optimization for ~10GB RAM."""

import gc
import logging
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from song2score.types import PartType, StemConfig

logger = logging.getLogger(__name__)


class DemucsSeparator:
    """Stem separator using Meta's Demucs - memory optimized for 10GB RAM."""

    # Available Demucs models (sorted by memory usage)
    MODELS = {
        "htdemucs_ft": "htdemucs_ft",  # Fine-tuned, lighter (default)
        "htdemucs": "htdemucs",  # Best quality, heavier
        "htdemucs_6s": "htdemucs_6s",  # 6 stems
        "htdemucs_6s_ft": "htdemucs_6s_ft",
        "mdx": "mdx",
        "mdx_extra": "mdx_extra",
        "mdx_q": "mdx_q",  # Quantized, lowest memory
        "htdemucs_light": "htdemucs_light",  # If available
    }

    # Stem names for 4-stem models
    STEMS_4 = ["vocals", "drums", "bass", "other"]

    # Stem names for 6-stem models
    STEMS_6 = ["vocals", "drums", "bass", "other", "guitar", "piano"]

    def __init__(
        self,
        model: str = "htdemucs_ft",  # Use lighter model by default
        device: Optional[str] = None,
        segment_length: float = 4.0,  # Reduced to 4s for lower memory
        shifts: int = 0,  # No shifts to save memory
        use_float16: bool = False,  # Half precision - disabled (MKL FFT doesn't support it on CPU)
        use_cli: bool = False,  # Use CLI subprocess for better memory isolation
        workers: int = 1,  # Number of parallel workers (1 for low memory, 2-4 for faster)
        max_parallel_segments: int = 2,  # Max segments to process in parallel
    ):
        """Initialize the Demucs separator.

        Args:
            model: Demucs model name (default: htdemucs_ft for memory efficiency)
            device: Device to use (cpu, cuda, mps)
            segment_length: Segment length in seconds (4s recommended for 10GB RAM)
            shifts: Number of predictions with random shifts (0 saves memory)
            use_float16: Use half precision (only works with GPU, MKL FFT doesn't support on CPU)
            use_cli: Use demucs CLI subprocess (isolates memory better)
            workers: Number of workers for demucs (1 recommended for low memory)
            max_parallel_segments: Max segments to process in parallel (2-4 for speed)
        """
        self.model = model
        self.device = device or self._auto_detect_device()
        self.segment_length = segment_length
        self.shifts = shifts
        # Only use float16 on GPU/CUDA (MKL FFT doesn't support half precision on CPU)
        self.use_float16 = use_float16 and self.device in ("cuda", "mps")
        self.use_cli = use_cli
        self.workers = workers
        self.max_parallel_segments = max_parallel_segments

        # Determine number of stems based on model
        self._num_stems = 6 if "6s" in model else 4
        self.stem_names = self.STEMS_6 if self._num_stems == 6 else self.STEMS_4

    def _auto_detect_device(self) -> str:
        """Auto-detect the best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @property
    def num_stems(self) -> int:
        """Return the number of stems for the current model."""
        return self._num_stems

    def separate(
        self,
        input_path: Path,
        output_dir: Path,
        parts: Optional[List[PartType]] = None,
    ) -> Dict[PartType, Path]:
        """Separate audio into stems - memory optimized for 10GB RAM.

        Processes the audio in segments and saves partial results.
        """
        # Use CLI subprocess for better memory isolation if requested
        if self.use_cli:
            return self._separate_with_cli(input_path, output_dir, parts)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # output_dir already includes "stems" from the pipeline
        stems_output_dir = output_dir

        # Get audio info first
        info = sf.info(str(input_path))
        sr = info.samplerate
        total_samples = info.frames

        # Calculate segment size
        segment_samples = int(self.segment_length * sr)
        num_segments = int(np.ceil(total_samples / segment_samples))

        logger.info(f"Processing {total_samples/sr:.1f}s audio in {num_segments} segments of {self.segment_length}s each")
        logger.info(f"Audio: {info.channels} channels, {sr} Hz, {total_samples/sr:.1f}s")
        logger.info(f"Memory optimization: float16={self.use_float16}, segment={self.segment_length}s")

        # Temporary directory for segment stems
        temp_segments_dir = stems_output_dir / ".segments"
        temp_segments_dir.mkdir(exist_ok=True)

        # Process each segment (with parallel processing if enabled)
        if self.max_parallel_segments > 1 and num_segments > 1:
            logger.info(f"Processing {num_segments} segments with up to {self.max_parallel_segments} parallel workers")
            self._process_segments_parallel(
                input_path, sr, segment_samples, num_segments,
                temp_segments_dir, total_samples
            )
        else:
            # Sequential processing (original behavior)
            for seg_idx in range(num_segments):
                start_sample = seg_idx * segment_samples
                end_sample = min(start_sample + segment_samples, total_samples)

                # Read this segment with dtype=float32 for processing
                segment_len = end_sample - start_sample
                start_frame = start_sample
                audio = sf.read(str(input_path), start=start_frame, frames=segment_len, always_2d=True, dtype='float32')[0]

                # Convert to (channels, samples) format for demucs
                # always_2d=True returns (samples, channels), so always transpose
                audio = audio.T  # -> (channels, samples)

                # Demucs expects stereo input (2 channels), convert mono to stereo if needed
                if audio.shape[0] == 1:
                    # Duplicate mono channel to create stereo
                    audio = np.repeat(audio, 2, axis=0)

                logger.info(f"Processing segment {seg_idx + 1}/{num_segments} ({start_sample/sr:.1f}s - {end_sample/sr:.1f}s)")

                # Process this segment
                segment_sources = self._process_segment(audio, sr)

                # Save each stem segment immediately (streaming)
                for i, stem_name in enumerate(self.stem_names):
                    stem_segment_dir = temp_segments_dir / stem_name
                    stem_segment_dir.mkdir(exist_ok=True)

                    stem_segment_file = stem_segment_dir / f"segment_{seg_idx:03d}.wav"

                    # Transpose back to (samples, channels) for saving
                    stem_audio = segment_sources[i].T

                    # Save segment
                    sf.write(str(stem_segment_file), stem_audio, sr)

                # Aggressive memory cleanup
                del audio, segment_sources, stem_audio
                gc.collect()

        # Stream concatenate segments for each stem (one at a time)
        logger.info("Concatenating segments...")

        stems: Dict[PartType, Path] = {}

        for stem_name in self.stem_names:
            try:
                part_type = PartType(stem_name)
                if parts is None or part_type in parts:
                    stem_file = stems_output_dir / f"{stem_name}.wav"
                    stem_segment_dir = temp_segments_dir / stem_name

                    # Get all segment files
                    segment_files = sorted(stem_segment_dir.glob("segment_*.wav"))

                    if segment_files:
                        # Stream concatenate (one stem at a time)
                        self._stream_concatenate_wav_files(segment_files, stem_file)
                        stems[part_type] = stem_file
                        logger.info(f"Created stem: {stem_file}")

                        # Cleanup this stem's segments immediately
                        for f in segment_files:
                            f.unlink()
                        stem_segment_dir.rmdir()
            except ValueError:
                pass

        # Clean up temp directory
        import shutil
        if temp_segments_dir.exists():
            shutil.rmtree(temp_segments_dir)

        logger.info(f"Extracted {len(stems)} stems: {list(stems.keys())}")

        # Final cleanup - unload model
        if hasattr(self, '_model'):
            del self._model
            gc.collect()

        return stems

    def _process_segment(self, segment: np.ndarray, sr: int) -> np.ndarray:
        """Process a single audio segment through demucs with float16 support.

        Args:
            segment: Audio segment (channels, samples)
            sr: Sample rate

        Returns:
            Separated sources (stems, channels, samples)
        """
        try:
            from demucs import pretrained
            from demucs.apply import apply_model
            import torch
        except ImportError:
            raise ImportError(
                "demucs package not installed. "
                "Use 'pip install demucs'."
            )

        # Convert to tensor and add batch dimension
        audio_tensor = torch.from_numpy(segment).float()
        audio_tensor = audio_tensor.unsqueeze(0)  # (channels, samples) -> (1, channels, samples)

        # Use half precision if enabled (CPU only - more stable)
        if self.use_float16:
            audio_tensor = audio_tensor.half()

        # Load model once with precision optimization
        if not hasattr(self, '_model'):
            logger.info(f"Loading demucs model: {self.model} (float16={self.use_float16})")
            self._model = pretrained.get_model(self.model)
            self._model = self._model.to(self.device)

            # Convert model to half precision if enabled
            if self.use_float16:
                self._model = self._model.half()

            self._model.eval()  # Set to eval mode for inference

        # Apply model with minimal memory footprint
        with torch.no_grad():
            sources = apply_model(
                self._model,
                audio_tensor,
                shifts=self.shifts,
                split=True,  # Enable splitting
                segment=float(self.segment_length),  # Process in segments
                device=torch.device(self.device) if self.device != "cpu" else None,
                progress=False,
            )

        # Convert back to numpy and cleanup
        if self.use_float16:
            sources = sources.float()  # Convert back to float32 before numpy

        # Remove batch dimension: (1, stems, channels, samples) -> (stems, channels, samples)
        sources = sources.squeeze(0)

        sources = sources.cpu().numpy()

        # Immediate cleanup
        del audio_tensor
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return sources

    def _process_segment_task(
        self,
        seg_idx: int,
        input_path: Path,
        sr: int,
        segment_samples: int,
        temp_segments_dir: Path,
        total_samples: int,
    ) -> Tuple[int, List[Path]]:
        """Process a single segment - used for parallel processing.

        Args:
            seg_idx: Segment index
            input_path: Input audio file path
            sr: Sample rate
            segment_samples: Number of samples per segment
            temp_segments_dir: Temporary directory for segment stems
            total_samples: Total samples in audio

        Returns:
            Tuple of (segment_index, list_of_stem_files)
        """
        start_sample = seg_idx * segment_samples
        end_sample = min(start_sample + segment_samples, total_samples)

        # Read this segment with dtype=float32 for processing
        segment_len = end_sample - start_sample
        start_frame = start_sample
        audio = sf.read(str(input_path), start=start_frame, frames=segment_len, always_2d=True, dtype='float32')[0]

        # Convert to (channels, samples) format for demucs
        audio = audio.T  # -> (channels, samples)

        # Demucs expects stereo input (2 channels), convert mono to stereo if needed
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)

        logger.info(f"Processing segment {seg_idx + 1} ({start_sample/sr:.1f}s - {end_sample/sr:.1f}s)")

        # Process this segment (load model for this thread)
        segment_sources = self._process_segment(audio, sr)

        # Save each stem segment immediately
        stem_files = []
        for i, stem_name in enumerate(self.stem_names):
            stem_segment_dir = temp_segments_dir / stem_name
            stem_segment_dir.mkdir(exist_ok=True)

            stem_segment_file = stem_segment_dir / f"segment_{seg_idx:03d}.wav"

            # Transpose back to (samples, channels) for saving
            stem_audio = segment_sources[i].T

            # Save segment
            sf.write(str(stem_segment_file), stem_audio, sr)
            stem_files.append(stem_segment_file)

        # Aggressive memory cleanup
        del audio, segment_sources, stem_audio
        gc.collect()

        return (seg_idx, stem_files)

    def _process_segments_parallel(
        self,
        input_path: Path,
        sr: int,
        segment_samples: int,
        num_segments: int,
        temp_segments_dir: Path,
        total_samples: int,
    ) -> None:
        """Process multiple segments in parallel using ThreadPoolExecutor.

        Args:
            input_path: Input audio file path
            sr: Sample rate
            segment_samples: Number of samples per segment
            num_segments: Total number of segments
            temp_segments_dir: Temporary directory for segment stems
            total_samples: Total samples in audio
        """
        completed_count = 0

        with ThreadPoolExecutor(max_workers=self.max_parallel_segments) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    self._process_segment_task,
                    seg_idx,
                    input_path,
                    sr,
                    segment_samples,
                    temp_segments_dir,
                    total_samples,
                ): seg_idx
                for seg_idx in range(num_segments)
            }

            # Process as they complete
            for future in as_completed(futures):
                seg_idx = futures[future]
                try:
                    future.result()
                    completed_count += 1
                    logger.info(f"Completed {completed_count}/{num_segments} segments")
                except Exception as e:
                    logger.error(f"Segment {seg_idx} failed: {e}")
                    raise

    def _separate_with_cli(
        self,
        input_path: Path,
        output_dir: Path,
        parts: Optional[List[PartType]] = None,
    ) -> Dict[PartType, Path]:
        """Use demucs CLI subprocess for better memory isolation.

        This runs demucs in a separate process, which frees memory when done.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # output_dir already includes "stems" from the pipeline
        stems_output_dir = output_dir

        input_path = Path(input_path)

        logger.info(f"Using demucs CLI for memory-efficient separation")

        # Build demucs command
        cmd = [
            "python", "-m", "demucs.separate",
            "--out", str(stems_output_dir),
            "-n", self.model,
            "--segment", str(self.segment_length),
            "--shifts", str(self.shifts),
            "--workers", str(self.workers),
        ]

        if self.device != "cpu":
            cmd.extend(["--device", self.device])

        cmd.append(str(input_path))

        logger.info(f"Running: {' '.join(cmd)}")

        # Run demucs in subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            logger.error(f"Demucs CLI failed: {result.stderr}")
            raise RuntimeError(f"Demucs separation failed: {result.stderr}")

        logger.info(result.stdout)

        # Find the output files (demucs creates subdirectories)
        # Expected pattern: stems/{model_name}/{filename}/{stem}.wav
        model_dir = stems_output_dir / self.model
        track_name = input_path.stem

        track_output_dir = model_dir / track_name

        stems: Dict[PartType, Path] = {}

        if track_output_dir.exists():
            for stem_name in self.stem_names:
                try:
                    part_type = PartType(stem_name)
                    if parts is None or part_type in parts:
                        stem_file = track_output_dir / f"{stem_name}.wav"
                        if stem_file.exists():
                            # Copy to our expected location
                            dest_file = stems_output_dir / f"{stem_name}.wav"
                            import shutil
                            shutil.copy2(stem_file, dest_file)
                            stems[part_type] = dest_file
                            logger.info(f"Found stem: {dest_file}")
                except ValueError:
                    pass

        # Cleanup demucs output directory
        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir)

        logger.info(f"Extracted {len(stems)} stems: {list(stems.keys())}")

        return stems

    def _stream_concatenate_wav_files(self, input_files: List[Path], output_file: Path) -> None:
        """Concatenate multiple WAV files using streaming (low memory).

        Processes files in batches instead of loading all at once.

        Args:
            input_files: List of input WAV files
            output_file: Output WAV file path
        """
        if not input_files:
            return

        # Get info from first file
        first_info = sf.info(str(input_files[0]))
        sr = first_info.samplerate

        # Process in batches to control memory (process 5 segment files at a time)
        batch_size = 5
        audio_batches = []

        for i in range(0, len(input_files), batch_size):
            batch = input_files[i:i + batch_size]

            # Read this batch
            batch_audio = []
            for wav_file in batch:
                audio = sf.read(str(wav_file), dtype='float32')[0]
                batch_audio.append(audio)

            # Concatenate batch
            batch_combined = np.concatenate(batch_audio, axis=0)
            audio_batches.append(batch_combined)

            # Clear batch data
            del batch_audio
            gc.collect()

        # Concatenate all batches and write final file
        final_audio = np.concatenate(audio_batches, axis=0)
        sf.write(str(output_file), final_audio, sr)

        # Cleanup
        del final_audio, audio_batches
        gc.collect()

    def _concatenate_wav_files(self, input_files: List[Path], output_file: Path) -> None:
        """Concatenate multiple WAV files into one (legacy method).

        Args:
            input_files: List of input WAV files
            output_file: Output WAV file path
        """
        if not input_files:
            return

        # Read all audio data and concatenate
        audio_segments = []
        sr = None
        channels = None

        for wav_file in input_files:
            info = sf.info(str(wav_file))
            audio = sf.read(str(wav_file), dtype='float32')

            if sr is None:
                sr = info.samplerate
                channels = info.channels
            audio_segments.append(audio[0])

        # Concatenate
        combined_audio = np.concatenate(audio_segments, axis=0)

        # Save combined audio
        sf.write(str(output_file), combined_audio, sr)

        # Cleanup
        del combined_audio, audio_segments
        gc.collect()

    def get_stem_count(self) -> int:
        """Return the number of stems this separator produces."""
        return self.num_stems

    def get_stem_names(self) -> List[str]:
        """Return the list of stem names this separator produces."""
        return self.stem_names
