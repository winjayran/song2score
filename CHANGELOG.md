# Changelog

All notable changes to song2score will be documented in this file.

## [0.5.0] - 2026-03-15

### Added
- Stem refinement module to clean up mixed audio in separated stems
  - Harmonic-Percussive Source Separation (HPSS) to separate melodic and percussive content
  - Frequency band filtering to isolate instrument-specific frequency ranges
  - New `--refine-stems` CLI option for `transcribe` and `score` commands
  - Specialized refinement functions for vocals (remove drum bleed) and bass (remove high frequencies)
- New `StemRefiner` class in `song2score/separation/refinement.py`
  - Configurable HPSS margin, frequency filtering, and harmonic/percussive masking
  - Per-instrument frequency range optimization
- New `--model` CLI option to specify Demucs model directly
- New `--remap-stems` CLI option for manual stem reassignment
- Improved vocals detection with specialized feature scoring
  - Vocal-specific feature analysis (spectral centroid, onset rate, MFCCs)
  - Distinguishes vocals from strings/guitar based on speech-like patterns
- CLI improvements: `--model` and `--remap-stems` options for `transcribe` and `score` commands

### Fixed
- **Critical**: Fixed phantom stem creation bug where reclassification would create invalid mappings like `strings -> guitar.wav`
- Stem reclassification is now conservative - only reclassifies "other" stem to avoid breaking valid assignments
- Improved vocals detection to reduce misclassification as strings
- Updated feature ranges to better distinguish between vocals and strings instruments
- Report generation now correctly maps stem types to file paths without phantom assignments

### Changed
- Pipeline now supports optional stem refinement via `refine_stems` parameter
- Stem refinement is applied after classification but before transcription
- `_verify_and_correct_stem_classification()` simplified to only reclassify "other" stem
- Improved classification feature ranges for vocals vs strings distinction
- Stem separation defaults: `--stems 4` uses `htdemucs_ft`, `--stems 6` uses `htdemucs_6s`

### Technical
- New `song2score/separation/refinement.py` module with HPSS and frequency filtering
- Updated `InstrumentClassifier._check_vocals_specific()` for improved vocals detection
- Updated `InstrumentClassifier.FEATURE_RANGES` with better vocals/strings distinction
- Updated Pipeline to include `_refine_stems()` method
- Updated CLI `transcribe` and `score` commands with `--refine-stems`, `--model`, `--remap-stems` options
- Documentation updates for stem refinement feature and manual stem remapping

## [0.4.0] - 2026-03-14

### Added
- Stem reclassification based on audio content analysis
  - Automatically corrects misclassified stems when confidence is high (>65%)
  - Uses InstrumentClassifier to detect actual instrument types in each stem
  - Helps when Demucs incorrectly assigns instruments (e.g., guitar in "other")
- Separate MusicXML export for individual instrument parts
  - New `--separate-parts` flag for `export` and `score` commands
  - Creates individual MusicXML files for each instrument part
- Separate PDF generation for each part
  - New `--separate-parts` flag for `render` command
  - Generate individual PDFs for each instrument part

### Fixed
- MIDI transcription errors for empty or silent stems
  - Added audio validation before transcription to detect empty/quiet/silent audio
  - Gracefully skips stems with insufficient content instead of failing
  - Better error messages for transcription failures
- Improved error handling in Basic Pitch transcription
  - Validates generated MIDI files have actual notes
  - Catches ValueError for empty audio and skips gracefully
  - Reports useful error messages when transcription fails

### Changed
- Stem reclassification is now enabled by default with confidence threshold
- Pipeline now auto-corrects stem classifications when confidence is high
- Transcription errors are logged as warnings instead of failing the entire pipeline

### Technical
- New `_has_sufficient_audio()` method in BasicPitchTranscriber for audio validation
- Updated `_verify_and_correct_stem_classification()` in Pipeline to support auto-correction
- New `export_separate_parts()` method in MusicXMLExporter
- Enhanced `_transcribe_stem()` in Pipeline to handle ValueError for empty stems
- Updated CLI commands: `export`, `score`, and `render` with `--separate-parts` option

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
