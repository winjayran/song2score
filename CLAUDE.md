# song2score - Implementation Notes

This document contains implementation notes, architecture decisions, and technical details for contributors.

> **Important**: Update this document whenever you make changes to the codebase. This helps maintain consistency and aids future contributors.

## Overview

**Goal**: Convert mixed audio songs to separated parts, MIDI, and sheet music (MusicXML + PDF).

**Core Pipeline**:
1. Audio Preprocessing (ffmpeg) → unified format
2. Stem Separation (Demucs) → vocals/drums/bass/other (+ guitar/piano with 6-stem)
3. Stem Reclassification (InstrumentClassifier) → correct misclassifications based on content
4. MIDI Transcription (Basic Pitch, Madmom) → per-part MIDI
5. MusicXML Export (music21) → score with re-orchestration (combined or separate parts)
6. PDF Rendering (MuseScore) → printable sheet music

## Architecture

### Module Overview

```
song2score/
├── __main__.py           # CLI entry point (typer)
├── types.py              # Pydantic models, enums, MIDI instruments
├── pipeline.py           # Main orchestrator (Pipeline, QuickTranscribe)
├── audio/
│   └── preprocess.py     # Audio format conversion with ffmpeg
├── separation/
│   ├── demucs.py         # Demucs stem separation (memory-optimized, parallel)
│   ├── strings.py        # Strings detection from "other" stem
│   ├── classifier.py     # Instrument classification using audio features
│   └── refinement.py     # Stem refinement to clean up mixed audio (HPSS, frequency filtering)
├── transcription/
│   ├── basic_pitch.py    # Basic Pitch (guitar, piano, violin, general)
│   └── drums.py          # Drum transcription (Madmom)
├── export/
│   └── musicxml.py       # MusicXML export with TAB support, separate staves
└── render/
    └── musescore.py      # PDF/PNG/SVG rendering via MuseScore
```

### Key Classes

| Class | Module | Responsibility |
|-------|--------|----------------|
| `Pipeline` | pipeline.py | Full orchestration of all steps |
| `QuickTranscribe` | pipeline.py | Shortcut for common workflows |
| `DemucsSeparator` | separation/demucs.py | Stem separation with memory & parallel optimization |
| `StringsSeparator` | separation/strings.py | Strings detection from spectral analysis |
| `InstrumentClassifier` | separation/classifier.py | Audio feature-based instrument classification |
| `StemRefiner` | separation/refinement.py | Stem refinement to clean up mixed audio |
| `BasicPitchTranscriber` | transcription/basic_pitch.py | General MIDI transcription |
| `GuitarTranscriber` | transcription/basic_pitch.py | Guitar-specific transcription |
| `PianoTranscriber` | transcription/basic_pitch.py | Piano-specific transcription |
| `ViolinTranscriber` | transcription/basic_pitch.py | Strings-specific transcription |
| `DrumTranscriber` | transcription/drums.py | Drum kit transcription |
| `MusicXMLExporter` | export/musicxml.py | MIDI → MusicXML with TAB, separate staves |
| `MuseScoreRenderer` | render/musescore.py | MusicXML → PDF/PNG/SVG |

## CLI Design

### Commands

```bash
# Transcribe: Audio → Stems + MIDI
song2score transcribe INPUT --out DIR [OPTIONS]

# Export: MIDI → MusicXML
song2score export MIDI_DIR --out DIR [OPTIONS]

# Score: Audio → Stems + MIDI + MusicXML (+ optional PDF)
song2score score INPUT --out DIR [OPTIONS]

# Render: MusicXML → PDF/PNG/SVG
song2score render MUSICXML --out FILE [OPTIONS]
```

### Key Options

| Command | Key Options |
|---------|-------------|
| `transcribe` | `--stems 4\|6`, `--model MODEL`, `--parts PARTS`, `--remap-stems MAP` |
| `export` | `--parts PARTS`, `--map MAP`, `--guitar-tab`, `--separate-parts`, `--title TITLE` |
| `score` | All transcribe options + `--map`, `--guitar-tab`, `--separate-parts`, `--pdf`, `--musescore PATH` |
| `render` | `--out FILE`, `--format pdf\|png\|svg`, `--resolution DPI`, `--separate-parts`, `--auto-install-musescore` |

**Note**: The `render` command uses `--out` (not `--output`) for the output file path.

## Stem Remapping (v0.4.0+)

**File**: `song2score/pipeline.py`

Demucs doesn't always perfectly separate instruments. Sometimes guitar ends up in "other", or drums leak into vocals. song2score provides two ways to handle this:

### Manual Stem Remapping

Use the `--remap-stems` CLI option to manually specify which stem file should be treated as which instrument:

```bash
# Example: If Demucs put guitar in "other" and drums in "vocals"
song2score transcribe input.mp3 --out output/ --remap-stems other=guitar,vocals=drums
```

This is useful when:
- The stem separation model consistently misclassifies a particular instrument
- You want to override the default stem assignments
- Working with specific songs where Demucs makes predictable errors

### Auto-Correction (Experimental)

The pipeline can automatically detect and correct stem misclassifications using audio feature analysis:

```python
stems = pipeline._verify_and_correct_stem_classification(
    stems,
    auto_correct=True,      # Enable auto-correction
    min_confidence=0.65,    # Minimum confidence for correction
)
```

**How it works**:
1. Each stem is analyzed using `InstrumentClassifier`
2. If confidence is high (≥65%), the stem is reassigned to the detected type
3. Conflicts are resolved by keeping the stem with the highest confidence

**Limitations**:
- Works best for clear, unmixed instrument sounds
- May misclassify heavily processed or mixed audio
- Cannot fix stems that contain multiple instruments (e.g., bass + piano mixed)

**Current Issues**: The auto-correction is limited by the quality of Demucs output. If a stem contains multiple instruments (e.g., bass.wav contains both bass AND piano), the classifier will only detect the dominant instrument.

## Stem Refinement (v0.5.0+)

**File**: `song2score/separation/refinement.py`

When Demucs output contains mixed audio (e.g., drums leaking into vocals, or high-frequency instruments in bass), the stem refinement module can help clean up the stems using audio processing techniques.

### Techniques Used

1. **Harmonic-Percussive Source Separation (HPSS)**
   - Separates audio into harmonic (melodic) and percussive (drums) components
   - For melodic instruments: keeps harmonic, attenuates percussive
   - For drums: keeps percussive, attenuates harmonic

2. **Frequency Band Filtering**
   - Each instrument has a characteristic frequency range
   - Applies bandpass filter to isolate relevant frequencies
   - Removes out-of-range content that may be from other instruments

### CLI Usage

```bash
# Apply stem refinement during transcription
song2score transcribe input.mp3 --out output/ --refine-stems

# Apply stem refinement during score generation
song2score score input.mp3 --out output/ --refine-stems --pdf
```

### Python API

```python
from song2score.separation.refinement import StemRefiner

refiner = StemRefiner(
    use_harmonic_mask=True,    # Apply HPSS for melodic instruments
    use_percussive_mask=True,  # Apply HPSS for drums
    use_frequency_filter=True, # Apply frequency band filtering
    margin=2.0,                # HPSS margin (higher = more aggressive)
)

# Refine a single stem
refined_path, metadata = refiner.refine_stem(
    stem_path="stems/vocals.wav",
    part_type=PartType.VOCALS,
    output_path="stems/vocals_refined.wav",
)
```

### Frequency Ranges

| Instrument | Frequency Range (Hz) |
|-----------|---------------------|
| Bass | 20 - 250 |
| Drums | 30 - 5000 |
| Guitar | 80 - 5000 |
| Piano | 28 - 4200 |
| Strings | 200 - 8000 |
| Vocals | 80 - 3500 |

### Limitations

- **Cannot fully separate mixed instruments**: If two instruments occupy the same frequency range and have similar harmonic/percussive characteristics, refinement won't fully separate them
- **May affect audio quality**: Aggressive filtering can remove desirable frequencies
- **Instrument-specific settings**: Different instruments may require different settings for optimal results

## Lazy Loading and Warning Suppression

**Important for CLI responsiveness**: The project uses lazy imports to avoid loading heavy dependencies (TensorFlow, Basic Pitch) at CLI startup.

### Implementation

**Files**: `song2score/__init__.py`, `song2score/__main__.py`, `song2score/pipeline.py`

**Purpose**: Avoid warnings and slow startup when running commands like `--version` or `--help`

**How it works**:
1. `__init__.py` - Suppresses dependency warnings before any imports
2. `__main__.py` - Lazy imports Pipeline and QuickTranscribe
3. `pipeline.py` - Lazy imports transcribers via `_get_transcribers()`
4. `basic_pitch.py` - Lazy imports Basic Pitch modules via `_import_basic_pitch()`

**Warning Suppression**:
```python
# Environment variables (set before imports)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # TensorFlow logging
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')  # oneDNN warnings

# Warning filters
warnings.filterwarnings('ignore', message='.*urllib3.*doesn\'t match.*')
warnings.filterwarnings('ignore', message='.*chardet.*doesn\'t match.*')
```

### When Adding New Heavy Dependencies

1. Add lazy import in the relevant module
2. Add warning suppression if the dependency emits warnings on import
3. Update this document

## Parallel Processing

**File**: `song2score/separation/demucs.py`, `song2score/pipeline.py`

song2score now supports parallel processing for faster execution:

### Stem Separation Parallelization

The `DemucsSeparator` can process multiple audio segments in parallel:

```python
separator = DemucsSeparator(
    model="htdemucs_ft",
    max_parallel_segments=2,  # Process 2 segments at once
)
stems = separator.separate(input_path, output_dir)
```

**Configuration**:
- `max_parallel_segments`: Number of segments to process simultaneously (1-4)
- Higher values = faster processing but more memory usage
- Recommended: 2 for ~10GB RAM, 4 for 16GB+ RAM

### MIDI Transcription Parallelization

The `Pipeline` can transcribe multiple stems in parallel:

```python
pipeline = Pipeline(
    output_dir=output_dir,
    parallel_transcription=True,
    max_transcription_workers=3,  # Transcribe 3 stems at once
)
```

**Benefits**:
- 2-4x faster transcription when processing multiple stems
- Each stem is transcribed independently
- Memory-efficient: each worker loads only its own transcriber

## Instrument Classification

**File**: `song2score/separation/classifier.py`

The new `InstrumentClassifier` uses audio feature analysis to identify instruments:

**Features Used**:
- Harmonic-percussive ratio (distinguishes drums from melodic instruments)
- Zero crossing rate (brightness/transients)
- Spectral centroid (frequency content)
- Attack and decay characteristics (temporal envelope)
- MFCCs (timbral characteristics)

**Supported Classes**:
- `DRUMS` - Percussive, transient-rich, low harmonic content
- `GUITAR` - Harmonic, plucked, moderate decay
- `PIANO` - Harmonic, fast attack, exponential decay
- `STRINGS` - Harmonic, slow attack, sustained with vibrato
- `VOCALS` - Formant structure, speech-like patterns
- `BASS` - Low frequency, harmonic

```python
from song2score.separation.classifier import InstrumentClassifier

classifier = InstrumentClassifier()
instrument_class, confidence, features = classifier.classify(audio, sr)

# Or classify a file
instrument_class, confidence, features = classifier.classify_file(audio_path)

# Or classify in segments
segments = classifier.classify_segments(audio_path)
# Returns: [(start, end, instrument_class, confidence), ...]
```

**Integration with Pipeline**:

The pipeline uses the classifier to automatically correct stem misclassifications:

```python
stems = pipeline._verify_and_correct_stem_classification(
    stems,
    auto_correct=True,      # Enable auto-correction
    min_confidence=0.65,    # Minimum confidence for correction
)
```

**Stem Reclassification** (v0.4.0+):

When `auto_correct=True`, the pipeline will:
1. Classify the actual content of each stem
2. If confidence is high enough (≥65%), reassign the stem to the correct type
3. Handle conflicts when multiple stems map to the same detected type

This helps when Demucs misassigns instruments (e.g., guitar content ends up in "other").

## Empty Stem Handling

**File**: `song2score/transcription/basic_pitch.py`

The transcription now validates audio content before processing to avoid errors with empty or silent stems:

**Validation Checks**:
- Minimum duration (default 0.5 seconds)
- RMS energy level (detects silent audio)
- Significant signal ratio (detects mostly-silent audio)

```python
def _has_sufficient_audio(self, audio: np.ndarray, sr: int, min_duration: float = 0.5) -> Tuple[bool, str]:
    """Check if audio has sufficient content for transcription."""
    # Returns (has_content, reason)
```

When a stem has insufficient content:
- The transcription is skipped with a warning
- Pipeline continues processing other stems
- Error is logged as "Skipped" rather than "Failed"

## Separate Parts Export

**File**: `song2score/export/musicxml.py`

The `MusicXMLExporter` can now create individual MusicXML files for each instrument part:

```python
exported_parts = exporter.export_separate_parts(
    midi_files=midi_files,
    output_dir=output_dir / "parts",
    title="My Song",
)
# Returns: {PartType.VOCALS: Path(.../vocals.musicxml), ...}
```

**CLI Usage**:
```bash
# Export separate MusicXML files for each part
song2score export midi_dir --out output --separate-parts

# Generate separate PDFs for each part
song2score score input.mp3 --out output --separate-parts --pdf
```

## Stem Separation

### Demucs Integration

**File**: `song2score/separation/demucs.py`

**Models**:
- `htdemucs_ft` - Fine-tuned, lighter (default)
- `htdemucs` - Best quality, heavier
- `htdemucs_6s` - 6 stems (adds guitar, piano)
- `mdx`, `mdx_extra`, `mdx_q` - Alternative models

**Memory Optimizations** (for ~10GB RAM):
1. Segment processing: 4-second segments
2. Streaming concatenation: Process 5 segments at a time
3. Aggressive GC: Explicit `gc.collect()` after operations
4. Float32 precision: MKL FFT doesn't support float16 on CPU
5. CLI subprocess option: `use_cli=True` for better memory isolation

**Mono Audio Handling**:
- Demucs expects stereo input (2 channels)
- Mono audio is automatically converted to stereo by duplicating the channel
- Audio shape handling: `sf.read()` returns (samples, channels), always transpose to (channels, samples)

**Usage**:
```python
separator = DemucsSeparator(
    model="htdemucs_ft",   # Default (lighter, ~2GB memory)
    segment_length=4.0,    # seconds
    shifts=0,              # no shifts for memory
    use_cli=False,         # or True for subprocess mode
)
stems = separator.separate(input_path, output_dir)
```

### Stem Names

- **4 stems**: `vocals`, `drums`, `bass`, `other`
- **6 stems**: `vocals`, `drums`, `bass`, `other`, `guitar`, `piano`

## MIDI Transcription

### Basic Pitch Integration

**File**: `song2score/transcription/basic_pitch.py`

**Specialized Transcribers**:
- `GuitarTranscriber` - Guitar-specific settings, TAB preparation
- `PianoTranscriber` - Piano-specific (sustain pedal, wide dynamic range)
- `ViolinTranscriber` - Strings-specific (vibrato, glissando)

**Parameters**:
- `confidence_threshold`: Minimum confidence for note detection (0-1)
- `minimum_note_length`: Minimum note duration in seconds
- `midi_tempo`: Default tempo for output MIDI

### Drum Transcription

**File**: `song2score/transcription/drums.py`

**Uses**: Madmom for beat tracking and drum event detection

**Status**: Has import issues with current Madmom version. Drums stem is still created for manual processing.

## MusicXML Export

### File Structure

**File**: `song2score/export/musicxml.py`

**Features**:
1. **Instrument Mapping**: Re-orchestrate parts (e.g., vocals → violin)
2. **Guitar TAB**: Generate tablature notation
3. **Clef Assignment**: Automatic clef selection per instrument
4. **Separate Staves**: Each instrument on its own staff with proper spacing
5. **Metadata**: Title, composer, encoding info
6. **Layout Control**: Proper page, system, and staff layout for PDF rendering

**Separate Staves Implementation**:

Each instrument is placed on a separate staff with:
- Unique part ID and name
- Proper system distance between staves
- Staff groups for instruments with multiple staves (piano grand staff)
- Page layout with proper margins for PDF rendering

```python
# Parts are automatically ordered: vocals, guitar, piano, bass, strings, drums, other
# Each part gets its own staff with proper spacing
```

**Layout Control**:

The exporter now adds comprehensive layout information:
- Page size: A4 (2100 x 2970 tenths)
- Margins: 100 tenths on all sides
- System distance: 80 (space between systems)
- Staff distance: 60 (space between staves in a system)
- Staff groups: Brace for piano, bracket for guitar

**Instrument Mapping**:
```python
# Map PartType to instrument name
instrument_map = {
    PartType.VOCALS: "violin",
    PartType.GUITAR: "acoustic_guitar_nylon",
    PartType.BASS: "contrabass",
}
```

**Guitar TAB**:
- Uses standard tuning: E A D G B E
- MIDI string numbers: [64, 59, 55, 50, 45, 40]
- Enabled via `--guitar-tab` flag

### music21 Integration

**Note**: music21 API has changed between versions. Current code uses:
- `converter.parse()` for MIDI import (not `stream.converter.parse()`)
- `clef.clefFromString(name)` for creating clefs (not `clef.Clef(name)`)
- Direct Part manipulation for instrument updates

**API Compatibility**:
```python
# Correct (v9+):
part.insert(0, clef.clefFromString("treble"))
part.insert(0, clef.TrebleClef())

# Incorrect (old API):
part.insert(0, clef.Clef("treble"))  # Raises TypeError
```

## PDF Rendering

### MuseScore Integration

**File**: `song2score/render/musescore.py`

**Executables searched**:
- Linux: `mscore`, `/usr/bin/mscore`
- macOS: `MuseScore.app`
- Windows: `MuseScore.exe`

**Auto-Install** (Linux only):
```python
renderer = MuseScoreRenderer(auto_install=True)
# Downloads MuseScore-Studio AppImage from GitHub releases
# Installs to ~/.local/bin/mscore
```

**Command Line Options** (version differences):
```bash
# PDF output (all versions)
mscore -o output.pdf input.musicxml

# PNG output with resolution
# MuseScore 2.x: -r for resolution
mscore -o output.png -r 300 input.musicxml

# MuseScore 3.x+: -T for trim, resolution via output extension
mscore -o output-300dpi.png input.musicxml
```

**PNG Output Notes**:
- MuseScore 2.x adds page suffixes to multi-page PNGs: `score-1.png`, `score-2.png`, etc.
- The renderer automatically detects and returns the first page when this occurs

## Type System

### PartType Enum

```python
class PartType(str, Enum):
    VOCALS = "vocals"
    DRUMS = "drums"
    BASS = "bass"
    GUITAR = "guitar"
    PIANO = "piano"
    STRINGS = "strings"
    OTHER = "other"
```

### Config Classes

All configs use Pydantic `BaseModel`:
- `StemConfig` - Stem separation settings
- `TranscriptionConfig` - Transcription parameters
- `ExportConfig` - MusicXML export settings
- `DeviceConfig` - CPU/GPU selection

### MIDI Instruments

`MIDI_INSTRUMENTS` dict maps instrument names to MIDI program numbers (0-127).

## Output Structure

```
output/
├── stems/               # Separated audio
│   ├── vocals.wav
│   ├── drums.wav
│   ├── bass.wav
│   └── other.wav
├── midi/                # Transcribed MIDI
│   ├── vocals.mid
│   ├── bass.mid
│   └── other.mid
├── musicxml/
│   └── score.musicxml   # Combined score
├── score.pdf            # If --pdf specified
└── report.json          # Processing report
```

## Dependencies

### Core
- `typer` - CLI framework
- `rich` - Terminal output
- `pydantic` - Data validation

### Audio
- `numpy` - Numerical operations
- `soundfile` - Audio I/O
- `librosa` - Audio analysis
- `ffmpeg-python` - ffmpeg wrapper

### MIDI
- `mido` - MIDI file I/O
- `pretty_midi` - MIDI manipulation

### Notation
- `music21` - Music notation

### ML Models
- `demucs` - Stem separation
- `basic-pitch` - Audio to MIDI
- `madmom` - Drum transcription
- `torch` - ML framework

## Known Issues

1. **Drum Transcription**: Madmom import fails with current version
   - Workaround: Drums stem is still created for manual use
   - Future: Update to compatible drum transcription

2. **Float16 on CPU**: MKL FFT doesn't support half precision
   - Solution: Disabled by default on CPU
   - GPU can use float16 for memory savings

3. **MuseScore Version Compatibility**: Different versions use different CLI options
   - MuseScore 2.x: `-r` for PNG resolution
   - MuseScore 3.x+: Different options for PNG resolution
   - Current code handles MuseScore 2.x correctly

4. **Dependency Warnings**: Some transitive dependencies emit warnings
   - Suppressed at import time for clean CLI output
   - See "Lazy Loading and Warning Suppression" section above

5. **Instrument Classification Accuracy**: The new classifier uses heuristic feature ranges
   - Works well for clear instrument sounds
   - May misclassify heavily processed or mixed audio
   - Confidence scores should be checked for critical applications

## Development Notes

### Version History

See `CHANGELOG.md` for detailed version history.

**Recent versions**:
- **v0.5.0** (2026-03-15): Stem refinement and classification improvements
  - Conservative stem reclassification (only "other" stem to avoid conflicts)
  - Stem refinement module (HPSS, frequency filtering) to clean up mixed audio
  - Improved vocals detection with specialized feature scoring
  - New `--refine-stems` CLI option
  - Fixed phantom stem creation bug
  - New `--model` and `--remap-stems` CLI options

- **v0.4.0** (2026-03-14): Stem reclassification and separate parts export
  - Automatic stem reclassification based on audio content analysis
  - Manual stem remapping via `--remap-stems` CLI option
  - Separate MusicXML/PDF export for individual instrument parts
  - Improved empty stem handling (skips gracefully instead of failing)
  - Better error handling in Basic Pitch transcription
  - New `--model` option to specify Demucs model directly

- **v0.3.0** (2026-03-14): Performance improvements, better layout
  - Parallel segment processing for stem separation (2-4x faster)
  - Parallel MIDI transcription for multiple stems
  - Instrument classifier using audio feature analysis (for logging/debugging)
  - Separate staves for different instruments in MusicXML/PDF output
  - Better PDF layout with proper staff spacing
  - Fixed README.md documentation (--output → --out for render)

- **v0.2.0** (2026-03-13): Warning suppression, lazy imports, bug fixes
  - Clean CLI startup (no TensorFlow/Basic Pitch warnings)
  - Fixed mono audio handling
  - Fixed music21 Clef API compatibility
  - Fixed MuseScore 2.x PNG rendering
  - Default model changed to `htdemucs_ft` (lower memory)

- **v0.1.0** (Initial): Core functionality
  - Stem separation, MIDI transcription, MusicXML/PDF export

### Adding a New Transcriber

1. Create class in `transcription/base.py` or extend existing
2. Implement `transcribe(audio_path, midi_path)` method
3. Register in `pipeline.py`
4. Add CLI option if needed

### Adding a New Instrument Mapping

1. Add to `MIDI_INSTRUMENTS` in `types.py`
2. Add clef mapping in `MusicXMLExporter.CLEF_ASSIGNMENTS`
3. Update documentation

### Memory Optimization Tips

1. Process audio in segments (4-6 seconds recommended for 10GB RAM)
2. Use streaming for file operations (don't load all at once)
3. Explicit `gc.collect()` after large operations
4. Use subprocess for isolation (`use_cli=True`)

### Keeping Documentation Updated

**IMPORTANT**: Whenever you make changes to the codebase:

1. **Update CLAUDE.md** - This is the primary technical reference
   - Document new modules, classes, or significant changes
   - Update API compatibility notes if using external libraries
   - Add any new configuration options or parameters

2. **Update CHANGELOG.md** - Track user-visible changes
   - Add new features under "Added"
   - Document bug fixes under "Fixed"
   - Note breaking changes under "Breaking"

3. **Update README.md** if user-facing behavior changes
   - New CLI options
   - Changed default behavior
   - New dependencies or system requirements

4. **Update .gitignore** for new temporary files
   - Add patterns for any new temp directories or cache files
   - Exclude test outputs and large generated files

5. **Update Version History** in CLAUDE.md
   - Add new version entries after releases
   - Document breaking changes for developers

6. **Track Current Issues and Update Plans** in CLAUDE.md
   - Document known limitations and bugs
   - Track planned improvements and features
   - Note areas that need refactoring or improvement

## Current Issues and Update Plans

### Known Issues (v0.5.0)

1. **Stem Separation Quality** (RESOLVED)
   - **Previous Issue**: Demucs doesn't perfectly separate instruments. Stems often contain mixed audio (e.g., bass.wav contains piano + strings, vocals.wav contains drums)
   - **Current Workarounds**:
     - Use `--remap-stems` to override stem assignments when consistent errors occur
     - Use `--refine-stems` to apply HPSS and frequency filtering to clean up mixed audio
   - **Status**: Stem refinement module added in v0.5.0 helps but cannot fully separate mixed instruments

2. **Instrument Classification Accuracy** (IMPROVED)
   - **Previous Issue**: The `InstrumentClassifier` uses heuristic feature ranges which may misclassify heavily processed audio
   - **Current Behavior**:
     - Improved vocals detection with specialized feature scoring
     - Classification is now conservative - only reclassifies "other" stem to avoid conflicts
     - Works well for clear instrument sounds, still struggles with heavily processed audio
   - **Planned Fix**: Train ML model on instrument datasets for better accuracy

3. **Drum Transcription** (UNRESOLVED)
   - **Issue**: Madmom import fails with current version
   - **Workaround**: Drums stem is still created for manual processing
   - **Planned Fix**: Update to compatible drum transcription or alternative library

4. **Vocals vs Strings Confusion** (IMPROVED)
   - **Previous Issue**: Vocals often misclassified as strings due to similar harmonic characteristics
   - **Current Behavior**: Specialized vocals detection with speech-like pattern analysis improves accuracy
   - **Remaining**: Can still confuse with very smooth string sections

### Update Plans (v0.6.0+)

1. **Improved Drum Transcription**
   - Replace Madmom with alternative library (e.g., pretty_midi drum patterns, beat tracking)
   - Implement drum pattern recognition from audio features

2. **Better Classification**
   - Collect training data for various instruments
   - Train custom classifier model using sklearn/pytorch
   - Add confidence-based warnings for uncertain classifications

3. **User Experience**
   - Add `--verbose` option for detailed logging
   - Better progress reporting during long operations
   - HTML report generation for analysis results

4. **Performance**
   - Adaptive parallel processing based on available memory
   - Faster stem separation with model optimization

## Testing

```bash
# Test transcription
song2score transcribe test.mp3 --out test_output/

# Test export
song2score export test_output/midi --out test_output/

# Test PDF (requires MuseScore)
song2score score test.mp3 --out test_output/ --pdf
```

## Future Work

1. **Improved Drum Transcription**: Replace Madmom or fix compatibility
2. **ML-based Instrument Classification**: Train a model on instrument datasets for better accuracy
3. **Tempo Detection**: Automatic tempo and time signature inference
4. **Batch Processing**: Multi-song support with parallel processing
5. **Web UI**: FastAPI + task queue
6. **Evaluation Metrics**: Confidence scoring, quality assessment
7. **Adaptive Parallel Processing**: Auto-tune worker count based on available memory
