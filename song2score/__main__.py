# Copyright (c) 2026 winjayran
# SPDX-License-Identifier: MIT
"""song2score CLI - Convert audio songs to sheet music."""

import logging
from pathlib import Path
from typing import Optional, List

import typer
from rich import print as rprint
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from song2score import __version__
from song2score.types import PartType
from song2score.pipeline import Pipeline, QuickTranscribe
from song2score.export.musicxml import MusicXMLExporter
from song2score.render.musescore import MuseScoreRenderer
from song2score.audio.preprocess import AudioPreprocessor

app = typer.Typer(
    name="song2score",
    help="Convert mixed audio songs to separated parts, MIDI, and sheet music.",
    add_completion=False,
)

console = Console()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Valid parts for CLI
VALID_PARTS = [p.value for p in PartType]

# Supported audio formats for help text
SUPPORTED_FORMATS_HELP = ", ".join(AudioPreprocessor.SUPPORTED_FORMATS[:5]) + ", ..."


def validate_audio_path(path: Path) -> Path:
    """Validate that the input file is a supported audio format."""
    if not AudioPreprocessor.is_supported_format(path):
        rprint(f"[bold red]Error:[/bold red] Unsupported audio format: '{path.suffix}'")
        rprint(f"[dim]Supported formats:[/dim] {SUPPORTED_FORMATS_HELP}")
        raise typer.Exit(1)
    return path


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        typer.echo(f"song2score {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Enable verbose output.",
    ),
) -> None:
    """song2score - Convert audio to sheet music.

    Transform mixed audio files (vocals + instruments) into:
    - Separated stems (vocals, drums, guitar, piano, strings, etc.)
    - MIDI files for each part
    - MusicXML scores with re-orchestration options
    - PDF sheet music (if MuseScore is installed)
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@app.command()
def transcribe(
    input: Path = typer.Argument(
        ...,
        exists=True,
        callback=validate_audio_path,
        help=f"Input audio file ({SUPPORTED_FORMATS_HELP})",
    ),
    out: Path = typer.Option(
        Path("output"),
        "--out",
        "-o",
        help="Output directory",
    ),
    parts: Optional[str] = typer.Option(
        None,
        "--parts",
        "-p",
        help=f"Comma-separated parts to process: {','.join(VALID_PARTS)}",
    ),
    stems: int = typer.Option(
        4,
        "--stems",
        "-s",
        help="Number of stems for separation (4 or 6)",
    ),
) -> None:
    """Separate and transcribe audio to MIDI.

    Separates the input audio into stems and transcribes each stem to MIDI.
    """
    rprint(f"[bold cyan]song2score[/bold cyan] - Transcribe mode")
    rprint(f"[dim]Input:[/dim] {input}")
    rprint(f"[dim]Output:[/dim] {out}")
    rprint(f"[dim]Stems:[/dim] {stems}")

    # Parse parts
    part_list: Optional[List[PartType]] = None
    if parts:
        part_names = [p.strip() for p in parts.split(",")]
        part_list = []
        for name in part_names:
            try:
                part_list.append(PartType(name))
            except ValueError:
                rprint(f"[yellow]Warning:[/yellow] Unknown part '{name}', skipping")

    with console.status("[bold green]Processing...") as status:
        status.update("Transcribing audio...")

        try:
            report = QuickTranscribe.transcribe(
                input_path=input,
                output_dir=out,
                parts=part_list,
                device="cpu",
            )

            # Show results
            rprint("\n[bold green]✓ Transcription complete![/bold green]")

            if report.stems_produced:
                rprint("\n[bold]Stems produced:[/bold]")
                for part, path in report.stems_produced.items():
                    rprint(f"  • {part.value}: {path}")

            if report.midi_produced:
                rprint("\n[bold]MIDI files:[/bold]")
                for part, path in report.midi_produced.items():
                    rprint(f"  • {part.value}: {path}")

            if report.errors:
                rprint("\n[bold red]Errors:[/bold red]")
                for error in report.errors:
                    rprint(f"  • {error}")

            if report.warnings:
                rprint("\n[bold yellow]Warnings:[/bold yellow]")
                for warning in report.warnings:
                    rprint(f"  • {warning}")

            rprint(f"\n[dim]Processing time: {report.processing_time_seconds:.2f}s[/dim]")

        except Exception as e:
            rprint(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(1)


@app.command()
def export(
    midi_dir: Path = typer.Argument(
        ...,
        exists=True,
        help="Directory containing MIDI files",
    ),
    out: Path = typer.Option(
        Path("output"),
        "--out",
        "-o",
        help="Output directory",
    ),
    parts: Optional[str] = typer.Option(
        None,
        "--parts",
        "-p",
        help=f"Comma-separated parts to include: {','.join(VALID_PARTS)}",
    ),
    map: Optional[str] = typer.Option(
        None,
        "--map",
        "-m",
        help="Instrument mapping (e.g., vocals=violin,guitar=acoustic_guitar)",
    ),
    guitar_tab: bool = typer.Option(
        False,
        "--guitar-tab",
        help="Enable guitar TAB output",
    ),
    title: str = typer.Option(
        "Transcribed Score",
        "--title",
        "-t",
        help="Score title",
    ),
) -> None:
    """Export MIDI files to MusicXML score.

    Combines MIDI files into a single MusicXML score with optional
    re-orchestration and guitar TAB.
    """
    rprint(f"[bold cyan]song2score[/bold cyan] - Export mode")

    # Parse parts
    part_list: List[PartType] = [
        PartType.VOCALS, PartType.GUITAR, PartType.PIANO, PartType.BASS
    ]
    if parts:
        part_names = [p.strip() for p in parts.split(",")]
        part_list = []
        for name in part_names:
            try:
                part_list.append(PartType(name))
            except ValueError:
                rprint(f"[yellow]Warning:[/yellow] Unknown part '{name}', skipping")

    # Parse instrument mapping
    instrument_map: dict = {}
    if map:
        for mapping in map.split(","):
            try:
                src, dst = mapping.strip().split("=")
                instrument_map[PartType(src)] = dst
            except ValueError:
                rprint(f"[yellow]Warning:[/yellow] Invalid mapping '{mapping}', skipping")

    # Find MIDI files
    midi_files = {}
    for part_type in part_list:
        midi_path = midi_dir / f"{part_type.value}.mid"
        if midi_path.exists():
            midi_files[part_type] = midi_path

    if not midi_files:
        rprint("[bold red]Error:[/bold red] No MIDI files found")
        rprint(f"  Looking for: {midi_dir}")
        raise typer.Exit(1)

    rprint(f"[dim]MIDI files found:[/dim] {len(midi_files)}")

    # Create exporter
    from song2score.types import ExportConfig

    config = ExportConfig(
        parts=part_list,
        instrument_map=instrument_map,
        guitar_tab=guitar_tab,
    )

    exporter = MusicXMLExporter(config)

    with console.status("[bold green]Exporting...") as status:
        status.update("Creating MusicXML score...")

        try:
            output_path = out / "musicxml" / "score.musicxml"
            output_path, metadata = exporter.export(
                midi_files=midi_files,
                output_path=output_path,
                title=title,
            )

            rprint("\n[bold green]✓ Export complete![/bold green]")
            rprint(f"[dim]Output:[/dim] {output_path}")

        except Exception as e:
            rprint(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(1)


@app.command()
def render(
    musicxml: Path = typer.Argument(
        ...,
        exists=True,
        help="Input MusicXML file",
    ),
    out: Path = typer.Option(
        None,
        "--out",
        "-o",
        help="Output path (defaults to same as input with .pdf extension)",
    ),
    format: str = typer.Option(
        "pdf",
        "--format",
        "-f",
        help="Output format (pdf, png, svg)",
    ),
    resolution: int = typer.Option(
        300,
        "--resolution",
        "-r",
        help="Resolution for PNG (DPI)",
    ),
    auto_install: bool = typer.Option(
        False,
        "--auto-install-musescore",
        help="Automatically download portable MuseScore if not found",
    ),
) -> None:
    """Render MusicXML to PDF/PNG using MuseScore.

    Requires MuseScore to be installed. Use --auto-install-musescore to automatically
    download a portable version (Linux only).
    """
    rprint(f"[bold cyan]song2score[/bold cyan] - Render mode")

    renderer = MuseScoreRenderer(auto_install=auto_install)

    if not renderer.is_available():
        rprint("[bold red]Error:[/bold red] MuseScore not found")
        rprint("  Install MuseScore to use the render command:")
        rprint("    Linux: sudo apt install musescore3")
        rprint("    Or use --auto-install-musescore to download portable version")
        rprint("    https://musescore.org/")
        raise typer.Exit(1)

    version = renderer.check_version()
    if version:
        rprint(f"[dim]MuseScore version:[/dim] {version}")

    with console.status("[bold green]Rendering...") as status:
        status.update(f"Rendering to {format.upper()}...")

        try:
            if out is None:
                out = musicxml.with_suffix(f".{format}")

            if format.lower() == "pdf":
                output_path, _ = renderer.render_to_pdf(musicxml, out)
            elif format.lower() == "png":
                output_path, _ = renderer.render_to_png(musicxml, out, resolution)
            elif format.lower() == "svg":
                output_path, _ = renderer.render_to_svg(musicxml, out)
            else:
                rprint(f"[bold red]Error:[/bold red] Unknown format '{format}'")
                raise typer.Exit(1)

            rprint("\n[bold green]✓ Render complete![/bold green]")
            rprint(f"[dim]Output:[/dim] {output_path}")

        except Exception as e:
            rprint(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(1)


@app.command()
def score(
    input: Path = typer.Argument(
        ...,
        exists=True,
        callback=validate_audio_path,
        help=f"Input audio file ({SUPPORTED_FORMATS_HELP})",
    ),
    out: Path = typer.Option(
        Path("output"),
        "--out",
        "-o",
        help="Output directory",
    ),
    parts: Optional[str] = typer.Option(
        None,
        "--parts",
        "-p",
        help=f"Comma-separated parts to include: {','.join(VALID_PARTS)}",
    ),
    map: Optional[str] = typer.Option(
        None,
        "--map",
        "-m",
        help="Instrument mapping (e.g., vocals=violin,guitar=acoustic_guitar)",
    ),
    guitar_tab: bool = typer.Option(
        False,
        "--guitar-tab",
        help="Enable guitar TAB output",
    ),
    title: str = typer.Option(
        "Transcribed Score",
        "--title",
        "-t",
        help="Score title",
    ),
    pdf: bool = typer.Option(
        False,
        "--pdf",
        help="Also render to PDF (requires MuseScore)",
    ),
    musescore_path: Optional[str] = typer.Option(
        None,
        "--musescore",
        help="Path to MuseScore executable (auto-detect if not specified)",
    ),
) -> None:
    """Complete pipeline: transcribe and export to MusicXML (+ optional PDF).

    This command combines transcription and export into one step.
    Use --pdf to also generate a PDF file (requires MuseScore).
    """
    rprint(f"[bold cyan]song2score[/bold cyan] - Score mode")
    rprint(f"[dim]Input:[/dim] {input}")
    rprint(f"[dim]Output:[/dim] {out}")
    if pdf:
        rprint(f"[dim]PDF output:[/dim] Enabled")

    # Parse parts
    part_list: Optional[List[PartType]] = None
    if parts:
        part_names = [p.strip() for p in parts.split(",")]
        part_list = []
        for name in part_names:
            try:
                part_list.append(PartType(name))
            except ValueError:
                rprint(f"[yellow]Warning:[/yellow] Unknown part '{name}', skipping")

    # Parse instrument mapping
    instrument_map: dict = {}
    if map:
        for mapping in map.split(","):
            try:
                src, dst = mapping.strip().split("=")
                instrument_map[PartType(src)] = dst
            except ValueError:
                rprint(f"[yellow]Warning:[/yellow] Invalid mapping '{mapping}', skipping")

    with console.status("[bold green]Processing...") as status:
        status.update("Transcribing and exporting...")

        try:
            report = QuickTranscribe.to_score(
                input_path=input,
                output_dir=out,
                parts=part_list,
                instrument_map=instrument_map,
                device="cpu",
            )

            # Show results
            rprint("\n[bold green]✓ Score generation complete![/bold green]")

            if report.midi_produced:
                rprint("\n[bold]MIDI files:[/bold]")
                for part, path in report.midi_produced.items():
                    rprint(f"  • {part.value}: {path}")

            if report.musicxml_produced:
                rprint(f"\n[bold]MusicXML:[/bold] {report.musicxml_produced}")

            # Try to render PDF
            if pdf and report.musicxml_produced:
                renderer = MuseScoreRenderer(executable_path=musescore_path, auto_install=True)
                if renderer.is_available():
                    status.update("Rendering to PDF...")
                    try:
                        pdf_path = out / "score.pdf"
                        pdf_path, _ = renderer.render_to_pdf(report.musicxml_produced, pdf_path)
                        rprint(f"[bold]PDF:[/bold] {pdf_path}")
                    except Exception as e:
                        rprint(f"[yellow]PDF rendering failed:[/yellow] {e}")
                        rprint("[dim]Install MuseScore to enable PDF rendering:[/dim]")
                        rprint("[dim]  https://musescore.org/en/download[/dim]")
                else:
                    rprint("[yellow]MuseScore not found - PDF rendering skipped[/yellow]")
                    rprint("[dim]Install MuseScore to use --pdf option:[/dim]")
                    rprint("[dim]  Linux: sudo apt install musescore3[/dim]")
                    rprint("[dim]  macOS: brew install --cask muse-score[/dim]")
                    rprint("[dim]  Windows: https://musescore.org/en/download[/dim]")
            elif pdf:
                rprint("[yellow]No MusicXML was generated - cannot render PDF[/yellow]")

            if report.errors:
                rprint("\n[bold red]Errors:[/bold red]")
                for error in report.errors:
                    rprint(f"  • {error}")

            if report.warnings:
                rprint("\n[bold yellow]Warnings:[/bold yellow]")
                for warning in report.warnings:
                    rprint(f"  • {warning}")

            rprint(f"\n[dim]Processing time: {report.processing_time_seconds:.2f}s[/dim]")

        except Exception as e:
            rprint(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(1)


def cli() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli()
