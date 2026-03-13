# song2score

Convert mixed audio songs (vocals + multiple instruments) into:
- **Separated stems** (vocals, drums, bass, guitar, piano, strings)
- **MIDI files** for each part
- **Sheet music** (MusicXML + PDF) with optional guitar TAB

> **Status**: Alpha - Working but expect to review/edit results for complex mixes.
> **Platform**: CPU-based (no GPU required), Linux/macOS/Windows

## Features

- **Input formats**: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.wma`, `.aiff`, and more
- **Stem Separation**:
  - 4 stems: `vocals`, `drums`, `bass`, `other`
  - 6 stems: adds `guitar`, `piano`
  - Strings detection from spectral analysis (violin, viola, cello, contrabass)
- **MIDI Transcription**:
  - Guitar: Specialized transcriber with string/fret detection
  - Piano: Specialized transcriber for piano
  - Violin/Strings: Specialized transcriber with vibrato detection
  - Drums: Full drum kit transcription (kick, snare, hi-hat, cymbals, toms)
  - Vocals/Bass/Other: General transcription with Basic Pitch
- **Sheet Music Export**:
  - MusicXML (editable in MuseScore, Sibelius, Finale)
  - PDF rendering (via MuseScore)
  - Re-orchestration (map parts to different instruments)
  - Guitar TAB support

## Requirements

- Python 3.10+
- ffmpeg (required for audio preprocessing)
- MuseScore (optional, for PDF rendering)

## Installation

### Quick Install

```bash
# Create conda environment
conda create -n song2score python=3.11
conda activate song2score

# Install ffmpeg
conda install -c conda-forge ffmpeg

# Install song2core with all dependencies
pip install -e ".[demucs,basic-pitch,madmom,torch]"

# Or install step by step
pip install -e .
pip install demucs basic-pitch
```

### Install MuseScore (Optional, for PDF)

```bash
# Linux
sudo apt install musescore3

# macOS
brew install --cask muse-score

# Windows
# Download from https://musescore.org/en/download
```

## Quick Start

### 1. Transcribe Audio to MIDI

```bash
# Basic transcription (4 stems)
song2score transcribe input.mp3 --out output/

# With 6-stem separation (includes guitar, piano)
song2score transcribe input.mp3 --out output/ --stems 6

# Select specific parts
song2score transcribe input.mp3 --out output/ --parts vocals,bass,guitar
```

**Outputs:**
```
output/
├── stems/           # Separated audio (vocals.wav, drums.wav, bass.wav, other.wav)
├── midi/            # MIDI files for each part
└── report.json      # Processing report
```

### 2. Export to MusicXML

```bash
# Export MIDI files to MusicXML
song2score export output/midi/ --out output/ \
  --parts vocals,bass,guitar,piano \
  --title "My Song"

# With instrument re-orchestration
song2score export output/midi/ --out output/ \
  --map vocals=violin bass=contrabass guitar=acoustic_guitar

# With guitar TAB
song2score export output/midi/ --out output/ --guitar-tab
```

**Outputs:**
```
output/
└── musicxml/
    └── score.musicxml    # Open in MuseScore, Sibelius, etc.
```

### 3. Generate PDF Sheet Music

```bash
# Generate MusicXML + PDF in one command
song2score score input.mp3 --out output/ --title "My Song" --pdf

# With custom parts and instruments
song2score score input.mp3 --out output/ \
  --title "My Song" \
  --parts vocals,guitar,piano,bass \
  --map vocals=violin bass=contrabass \
  --guitar-tab \
  --pdf
```

**Outputs:**
```
output/
├── stems/
├── midi/
├── musicxml/score.musicxml
└── score.pdf             # Printable sheet music
```

### 4. Render MusicXML to PDF

```bash
# Render existing MusicXML to PDF
song2score render output/musicxml/score.musicxml --out score.pdf

# Render to PNG
song2score render output/musicxml/score.musicxml --out score.png --format png

# Auto-install portable MuseScore if not found (Linux)
song2score render output/musicxml/score.musicxml --out score.pdf --auto-install-musescore
```

## CLI Commands

### `song2score transcribe`

Separate audio into stems and transcribe to MIDI.

```bash
song2score transcribe INPUT --out DIR [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--stems 4\|6` | Number of stems for separation (default: 4) |
| `--parts PARTS` | Comma-separated parts to process |
| `--out DIR` | Output directory |

### `song2score export`

Export MIDI files to MusicXML.

```bash
song2score export MIDI_DIR --out DIR [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--parts PARTS` | Parts to include in score |
| `--map MAP` | Instrument mapping (e.g., `vocals=violin,bass=contrabass`) |
| `--guitar-tab` | Enable guitar TAB output |
| `--title TITLE` | Score title |

### `song2score score`

Complete pipeline: transcribe and export in one step.

```bash
song2score score INPUT --out DIR [OPTIONS]
```

All `transcribe` options plus:
| Option | Description |
|--------|-------------|
| `--map MAP` | Instrument re-orchestration mapping |
| `--guitar-tab` | Enable guitar TAB |
| `--pdf` | Also render to PDF |
| `--musescore PATH` | Path to MuseScore executable |

### `song2score render`

Render MusicXML to PDF/PNG/SVG.

```bash
song2score render MUSICXML --out FILE [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--format pdf\|png\|svg` | Output format (default: pdf) |
| `--resolution DPI` | Resolution for PNG (default: 300) |
| `--auto-install-musescore` | Auto-download portable MuseScore (Linux) |

## Supported Parts

| Part      | Detection | Transcription | TAB Support |
|-----------|-----------|---------------|-------------|
| Vocals    | ✓ (4/6)   | ✓             | -           |
| Drums     | ✓ (4/6)   | ✓ (full kit)  | -           |
| Bass      | ✓ (4/6)   | ✓             | -           |
| Guitar    | ✓ (6)     | ✓             | ✓           |
| Piano     | ✓ (6)     | ✓             | -           |
| Strings*  | ✓ (spec.) | ✓             | -           |
| Other     | ✓ (4/6)   | ✓             | -           |

*Strings detected via spectral analysis on "other" stem

## Instrument Mapping

Map any part to a different output instrument:

```bash
--map vocals=violin,guitar=acoustic_guitar,piano=marimba,bass=contrabass
```

Available instruments:
- **Strings**: `violin`, `viola`, `cello`, `contrabass`
- **Guitar**: `acoustic_guitar_nylon`, `acoustic_guitar_steel`, `electric_guitar_jazz`, `electric_guitar_clean`, etc.
- **Piano**: `acoustic_grand_piano`, `bright_acoustic_piano`, `electric_grand_piano`, etc.
- **Bass**: `electric_bass_finger`, `electric_bass_pick`, `fretless_bass`, `slap_bass_1`, etc.
- See `MIDI_INSTRUMENTS` in `song2score/types.py` for full list

## Project Structure

```
song2score/
├── song2score/
│   ├── __main__.py           # CLI entry point
│   ├── types.py              # Pydantic models and enums
│   ├── pipeline.py           # Main orchestrator
│   ├── audio/
│   │   └── preprocess.py     # Audio preprocessing with ffmpeg
│   ├── separation/
│   │   ├── demucs.py         # Demucs stem separation (memory-optimized)
│   │   └── strings.py        # Strings detection/separation
│   ├── transcription/
│   │   ├── basic_pitch.py    # Basic Pitch (guitar, piano, violin, general)
│   │   └── drums.py          # Drum transcription (Madmom)
│   ├── export/
│   │   └── musicxml.py       # MusicXML export with TAB
│   └── render/
│       └── musescore.py      # PDF rendering via MuseScore
├── scripts/
│   └── install_musescore.sh  # Install portable MuseScore
├── pyproject.toml
├── CLAUDE.md                 # Implementation notes
└── README.md
```

## Memory Optimization

The `DemucsSeparator` is optimized for systems with ~10GB RAM:

- **Segment processing**: Audio processed in 4-second segments
- **Streaming concatenation**: Temporary files processed in batches
- **Float32 precision**: Uses standard precision (MKL FFT doesn't support float16 on CPU)
- **Aggressive GC**: Explicit garbage collection after each operation
- **CLI subprocess option**: Better memory isolation when using `--use-cli`

## Roadmap

- [x] Guitar transcription with TAB support
- [x] Piano transcription
- [x] Violin/strings transcription
- [x] Drum transcription (full kit)
- [x] MusicXML export with re-orchestration
- [x] PDF rendering (MuseScore integration)
- [x] Multiple audio format support
- [ ] Improved strings separation (iterative refinement)
- [ ] Web UI (FastAPI) + job queue
- [ ] Evaluation metrics and confidence scoring
- [ ] Multi-song batch processing
- [ ] Tempo detection and time signature inference

## Troubleshooting

### "MuseScore not found"
```bash
# Install MuseScore
sudo apt install musescore3  # Linux
brew install --cask muse-score  # macOS

# Or use auto-install (Linux only)
song2score score input.mp3 --pdf --auto-install-musescore
```

### "MKL FFT doesn't support Half precision"
This is expected - float16 is disabled on CPU due to MKL limitations.

### Drum transcription fails
Madmom may have compatibility issues. Drums stem is still created for manual processing.

## License

MIT

## Acknowledgments

- [Demucs](https://github.com/facebookresearch/demucs) - Stem separation
- [Basic Pitch](https://github.com/spotify/basic-pitch) - Audio to MIDI transcription
- [Madmom](https://github.com/CPJKU/madmom) - Drum transcription
- [music21](https://github.com/cuthbertLab/music21) - Music notation
- [MuseScore](https://musescore.org/) - PDF rendering
