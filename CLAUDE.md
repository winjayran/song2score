# song2score - Implementation Notes

This document contains implementation notes, architecture decisions, and technical details for contributors.

## Overview

**Goal**: Convert mixed audio songs to separated parts, MIDI, and sheet music (MusicXML + PDF).

**Core Pipeline**:
1. Audio Preprocessing (ffmpeg) → unified format
2. Stem Separation (Demucs) → vocals/drums/bass/other (+ guitar/piano with 6-stem)
3. MIDI Transcription (Basic Pitch, Madmom) → per-part MIDI
4. MusicXML Export (music21) → score with re-orchestration
5. PDF Rendering (MuseScore) → printable sheet music

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
│   ├── demucs.py         # Demucs stem separation (memory-optimized)
│   └── strings.py        # Strings detection from "other" stem
├── transcription/
│   ├── basic_pitch.py    # Basic Pitch (guitar, piano, violin, general)
│   └── drums.py          # Drum transcription (Madmom)
├── export/
│   └── musicxml.py       # MusicXML export with TAB support
└── render/
    └── musescore.py      # PDF/PNG/SVG rendering via MuseScore
```

### Key Classes

| Class | Module | Responsibility |
|-------|--------|----------------|
| `Pipeline` | pipeline.py | Full orchestration of all steps |
| `QuickTranscribe` | pipeline.py | Shortcut for common workflows |
| `DemucsSeparator` | separation/demucs.py | Stem separation with memory optimization |
| `StringsSeparator` | separation/strings.py | Strings detection from spectral analysis |
| `BasicPitchTranscriber` | transcription/basic_pitch.py | General MIDI transcription |
| `GuitarTranscriber` | transcription/basic_pitch.py | Guitar-specific transcription |
| `PianoTranscriber` | transcription/basic_pitch.py | Piano-specific transcription |
| `ViolinTranscriber` | transcription/basic_pitch.py | Strings-specific transcription |
| `DrumTranscriber` | transcription/drums.py | Drum kit transcription |
| `MusicXMLExporter` | export/musicxml.py | MIDI → MusicXML with TAB |
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
| `transcribe` | `--stems 4\|6`, `--parts PARTS` |
| `export` | `--parts PARTS`, `--map MAP`, `--guitar-tab`, `--title TITLE` |
| `score` | All transcribe options + `--map`, `--guitar-tab`, `--pdf`, `--musescore PATH` |
| `render` | `--format pdf\|png\|svg`, `--resolution DPI`, `--auto-install-musescore` |

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

**Usage**:
```python
separator = DemucsSeparator(
    model="htdemucs_ft",
    segment_length=4.0,  # seconds
    shifts=0,            # no shifts for memory
    use_cli=False,       # or True for subprocess mode
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
4. **Metadata**: Title, composer, encoding info

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
- Direct Part manipulation for instrument updates

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

**Command Line**:
```bash
mscore -o output.pdf input.musicxml
```

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

3. **music21 API**: Various versions have different APIs
   - Current: Uses `converter.parse()` directly
   - Fixed import: Added `converter` to imports

## Development Notes

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
2. **Strings Separation**: Better iterative refinement
3. **Tempo Detection**: Automatic tempo and time signature inference
4. **Batch Processing**: Multi-song support
5. **Web UI**: FastAPI + task queue
6. **Evaluation Metrics**: Confidence scoring, quality assessment
