# Copyright (c) 2026 winjayran
# SPDX-License-Identifier: MIT
"""MuseScore rendering module for PDF/PNG output.

This module handles rendering MusicXML files to PDF or PNG using MuseScore.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, List, TYPE_CHECKING

from song2score.types import ProcessingReport

logger = logging.getLogger(__name__)


class MuseScoreRenderer:
    """Render MusicXML to PDF/PNG using MuseScore CLI."""

    # Common MuseScore executable names
    MUSESCORE_EXECUTABLES = [
        "mscore",           # Linux
        "MuseScore",        # macOS
        "mscore.exe",       # Windows
        "MuseScore.exe",    # Windows (alternative)
    ]

    # Common installation paths
    INSTALLATION_PATHS = {
        "linux": [
            "/usr/bin/mscore",
            "/usr/local/bin/mscore",
            "/usr/bin/MuseScore",
        ],
        "darwin": [
            "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
            "/Applications/MuseScore 3.app/Contents/MacOS/mscore",
            "/Applications/MuseScore.app/Contents/MacOS/mscore",
        ],
        "win32": [
            "C:\\Program Files\\MuseScore 4\\bin\\MuseScore.exe",
            "C:\\Program Files\\MuseScore 3\\bin\\MuseScore.exe",
            "C:\\Program Files (x86)\\MuseScore 4\\bin\\MuseScore.exe",
        ],
    }

    def __init__(self, executable_path: Optional[str] = None, auto_install: bool = False):
        """Initialize the MuseScore renderer.

        Args:
            executable_path: Path to MuseScore executable (auto-detect if None)
            auto_install: If True, attempt to download portable MuseScore if not found
        """
        self.executable_path = executable_path or self._find_musescore()
        self.available = self.executable_path is not None

        if not self.available and auto_install:
            logger.info("MuseScore not found, attempting to download portable version...")
            self.executable_path = self._install_portable()
            self.available = self.executable_path is not None

        if not self.available:
            logger.warning("MuseScore not found, PDF rendering will not be available")
            logger.info("To install MuseScore manually:")
            logger.info("  Linux: sudo apt install musescore3")
            logger.info("  Or run: bash scripts/install_musescore.sh")

    def _find_musescore(self) -> Optional[str]:
        """Find MuseScore executable.

        Returns:
            Path to MuseScore executable or None if not found
        """
        import sys
        import shutil

        # Check if mscore is in PATH
        for name in self.MUSESCORE_EXECUTABLES:
            path = shutil.which(name)
            if path:
                return path

        # Check platform-specific installation paths
        platform = sys.platform
        if platform in self.INSTALLATION_PATHS:
            for path_str in self.INSTALLATION_PATHS[platform]:
                path = Path(path_str)
                if path.exists():
                    return str(path)

        return None

    def _install_portable(self) -> Optional[str]:
        """Download and install portable MuseScore AppImage.

        Returns:
            Path to installed MuseScore or None if installation failed
        """
        import sys
        import os
        import json
        import tempfile
        import urllib.request
        import platform

        # Only support Linux for AppImage
        if sys.platform != "linux":
            logger.warning("Auto-install only supported on Linux")
            return None
        arch = platform.machine()
        arch_suffix = "x86_64" if arch == "x86_64" else "aarch64"

        install_dir = Path.home() / ".local" / "bin"
        install_dir.mkdir(parents=True, exist_ok=True)

        # Get latest version from GitHub API
        try:
            with urllib.request.urlopen("https://api.github.com/repos/musescore/MuseScore/releases/latest") as response:
                release = json.loads(response.read())
                version = release["tag_name"].lstrip("v")

                # Find the correct AppImage
                appimage_name = None
                for asset in release["assets"]:
                    name = asset["name"]
                    if f"MuseScore-Studio-{version}" in name and arch_suffix in name and name.endswith(".AppImage"):
                        appimage_name = name
                        download_url = asset["browser_download_url"]
                        break

                if not appimage_name:
                    logger.warning(f"Could not find MuseScore AppImage for {arch_suffix}")
                    return None

                appimage_path = install_dir / appimage_name

                if appimage_path.exists():
                    logger.info(f"Portable MuseScore already exists at {appimage_path}")
                    return str(appimage_path)

                # Download
                logger.info(f"Downloading MuseScore {version} for {arch_suffix}...")
                logger.info(f"URL: {download_url}")

                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    def report_progress(block_num, block_size, total_size):
                        downloaded = block_num * block_size
                        if total_size > 0:
                            pct = min(100, downloaded * 100 // total_size)
                            if downloaded % (10 * 1024 * 1024) == 0 or pct == 100:
                                logger.info(f"Downloaded: {pct}%")

                    urllib.request.urlretrieve(download_url, tmp.name, reporthook=report_progress)
                    tmp_path = Path(tmp.name)

                # Move to install location
                tmp_path.rename(appimage_path)
                appimage_path.chmod(0o755)

                # Create symlink
                mscore_link = install_dir / "mscore"
                if mscore_link.exists():
                    mscore_link.unlink()
                try:
                    mscore_link.symlink_to(appimage_path)
                except OSError:
                    # Symlink might fail if file exists, try to remove and recreate
                    if mscore_link.is_symlink() or mscore_link.exists():
                        mscore_link.unlink()
                    mscore_link.symlink_to(appimage_path)

                logger.info(f"Installed MuseScore to {appimage_path}")
                return str(appimage_path)

        except Exception as e:
            logger.error(f"Failed to install portable MuseScore: {e}")
            return None

    def render(
        self,
        musicxml_path: Path,
        output_path: Path,
        output_format: str = "pdf",
        resolution: int = 300,
    ) -> Tuple[Path, Dict]:
        """Render MusicXML to PDF or PNG.

        Args:
            musicxml_path: Input MusicXML file path
            output_path: Output file path
            output_format: Output format (pdf or png)
            resolution: Resolution for PNG (DPI)

        Returns:
            Tuple of (output_path, metadata_dict)

        Raises:
            RuntimeError: If MuseScore is not available or rendering fails
        """
        if not self.available:
            raise RuntimeError("MuseScore is not available for rendering")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            self.executable_path,
            "-o", str(output_path),
        ]

        # Add format option for PNG
        if output_format.lower() == "png":
            cmd.extend(["-T", str(resolution)])

        # Input file
        cmd.append(str(musicxml_path))

        # Run MuseScore
        logger.info(f"Running MuseScore: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,  # 1 minute timeout
            )

            if result.returncode != 0:
                logger.error(f"MuseScore error: {result.stderr}")
                raise RuntimeError(f"MuseScore rendering failed: {result.stderr}")

            # Check if output was created
            if not output_path.exists():
                # MuseScore might have created a file with different extension
                # Check for .pdf if we requested .png, or vice versa
                alt_extensions = ["pdf", "png", "svg", "mxl"]
                found = False

                for ext in alt_extensions:
                    alt_path = output_path.with_suffix(f".{ext}")
                    if alt_path.exists():
                        output_path = alt_path
                        found = True
                        break

                if not found:
                    raise RuntimeError(f"MuseScore did not create output file")

            metadata = {
                "format": output_format,
                "resolution": resolution if output_format == "png" else None,
                "executable": self.executable_path,
            }

            logger.info(f"Rendered to {output_path}")

            return output_path, metadata

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"MuseScore rendering timed out after 60 seconds")

    def render_to_pdf(
        self,
        musicxml_path: Path,
        output_path: Optional[Path] = None,
    ) -> Tuple[Path, Dict]:
        """Render MusicXML to PDF.

        Args:
            musicxml_path: Input MusicXML file path
            output_path: Output PDF path (defaults to same as input with .pdf extension)

        Returns:
            Tuple of (output_path, metadata_dict)
        """
        if output_path is None:
            output_path = musicxml_path.with_suffix(".pdf")

        return self.render(musicxml_path, output_path, "pdf")

    def render_to_png(
        self,
        musicxml_path: Path,
        output_path: Optional[Path] = None,
        resolution: int = 300,
    ) -> Tuple[Path, Dict]:
        """Render MusicXML to PNG.

        Args:
            musicxml_path: Input MusicXML file path
            output_path: Output PNG path (defaults to same as input with .png extension)
            resolution: Resolution in DPI

        Returns:
            Tuple of (output_path, metadata_dict)
        """
        if output_path is None:
            output_path = musicxml_path.with_suffix(".png")

        return self.render(musicxml_path, output_path, "png", resolution)

    def render_to_svg(
        self,
        musicxml_path: Path,
        output_path: Optional[Path] = None,
    ) -> Tuple[Path, Dict]:
        """Render MusicXML to SVG.

        Args:
            musicxml_path: Input MusicXML file path
            output_path: Output SVG path (defaults to same as input with .svg extension)

        Returns:
            Tuple of (output_path, metadata_dict)
        """
        if output_path is None:
            output_path = musicxml_path.with_suffix(".svg")

        return self.render(musicxml_path, output_path, "svg")

    def batch_render(
        self,
        musicxml_paths: list[Path],
        output_dir: Path,
        output_format: str = "pdf",
    ) -> list[Tuple[Path, Dict]]:
        """Render multiple MusicXML files.

        Args:
            musicxml_paths: List of input MusicXML file paths
            output_dir: Output directory
            output_format: Output format (pdf or png)

        Returns:
            List of (output_path, metadata_dict) tuples
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        for musicxml_path in musicxml_paths:
            output_path = output_dir / f"{musicxml_path.stem}.{output_format}"

            try:
                result = self.render(musicxml_path, output_path, output_format)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to render {musicxml_path}: {e}")
                results.append((None, {"error": str(e)}))

        return results

    def check_version(self) -> Optional[str]:
        """Get MuseScore version.

        Returns:
            Version string or None if MuseScore not available
        """
        if not self.available:
            return None

        try:
            result = subprocess.run(
                [self.executable_path, "-v"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            return result.stdout.strip()

        except Exception as e:
            logger.warning(f"Could not get MuseScore version: {e}")
            return None

    def is_available(self) -> bool:
        """Check if MuseScore is available.

        Returns:
            True if MuseScore executable was found
        """
        return self.available
