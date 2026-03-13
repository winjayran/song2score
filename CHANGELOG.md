# Changelog

All notable changes to song2score will be documented in this file.

## [0.3.0] - 2026-03-14

### Added
- Parallel stem segment processing using ThreadPoolExecutor for 2-4x faster separation
- Parallel MIDI transcription for multiple stems
- Instrument classifier using audio feature analysis (harmonic ratio, spectral features, attack/decay)
- Stem content logging - logs detected instrument types for each stem
- Separate staves for different instruments in MusicXML output
- Improved PDF layout with proper staff spacing and system layout
- Staff group support for piano (grand staff) and guitar (standard + TAB)

### Changed
- Parts in MusicXML are now ordered: vocals, guitar, piano, bass, strings, drums, other
- Each part gets unique ID and proper part naming for better PDF rendering
- Pipeline logs stem content classifications for debugging (does not reclassify)

### Fixed
- Fixed README.md render command documentation (`--output` → `--out`)
- Fixed stem reclassification issue that was causing transcription errors

### Performance
- 2-4x faster stem separation with parallel segment processing
- Parallel transcription reduces total processing time when multiple stems are present
- Memory-efficient parallel processing with configurable worker limits

### Technical
- New `InstrumentClassifier` class in `song2score/separation/classifier.py`
- Updated `DemucsSeparator` with `max_parallel_segments` parameter
- Updated `Pipeline` with `parallel_transcription` and `max_transcription_workers` parameters
- Enhanced `MusicXMLExporter._add_layout()` for proper staff layout
- Enhanced `MusicXMLExporter.export()` with part ordering and metadata



## [0.2.0] - 2026-03-13

### Added
- Lazy import system for transcribers to avoid loading TensorFlow/Basic Pitch at CLI startup
- Warning suppression for transitive dependencies (TensorFlow, basic-pitch, requests/urllib3)
- Proper handling for mono audio input (auto-convert to stereo for demucs)

### Changed
- **Breaking:** Default demucs model changed from `htdemucs` to `htdemucs_ft` for lower memory usage
- Updated music21 Clef API usage for compatibility (`clef.clefFromString()` instead of `clef.Clef()`)
- Improved MuseScore 2.x compatibility for PNG rendering (use `-r` instead of `-T` for resolution)

### Fixed
- Fixed audio shape transposition bug - now correctly handles both mono and stereo input
- Fixed PNG rendering fallback to detect MuseScore's page suffix naming (e.g., `score-1.png`)
- Fixed memory allocation error when processing mono audio files
- Fixed MIDI to MusicXML export with updated music21 API

### Technical
- Version bump from 0.1.0 to 0.2.0
- All CLI commands now run without warnings on startup:
  - `song2score transcribe` - Stem separation + MIDI transcription
  - `song2score export` - MIDI to MusicXML
  - `song2score score` - Full pipeline (transcribe + export)
  - `song2score render` - MusicXML to PDF/PNG/SVG

## [0.1.0] - Initial Release

### Features
- Audio stem separation using Demucs (4 or 6 stems)
- MIDI transcription using Basic Pitch
- Specialized transcribers for guitar, piano, and violin
- MusicXML export with instrument re-orchestration
- Guitar TAB support
- PDF rendering via MuseScore
- Multiple audio format support (mp3, wav, flac, ogg, m4a, etc.)
