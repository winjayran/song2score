#!/bin/bash
# Download and setup portable MuseScore for Linux

set -e

MUSESCORE_VERSION="4.4.2"
MUSESCORE_APPIMAGE="MuseScore-${MUSESCORE_VERSION}.x86_64.AppImage"
INSTALL_DIR="$HOME/.local/bin"
DOWNLOAD_URL="https://github.com/musescore/MuseScore/releases/download/v${MUSESCORE_VERSION}/${MUSESCORE_APPIMAGE}"

echo "Downloading portable MuseScore..."
mkdir -p "$INSTALL_DIR"

if [ ! -f "$INSTALL_DIR/$MUSESCORE_APPIMAGE" ]; then
    wget -O "$INSTALL_DIR/$MUSESCORE_APPIMAGE" "$DOWNLOAD_URL"
    chmod +x "$INSTALL_DIR/$MUSESCORE_APPIMAGE"
fi

# Create symlink
ln -sf "$INSTALL_DIR/$MUSESCORE_APPIMAGE" "$INSTALL_DIR/mscore"

echo "MuseScore installed to: $INSTALL_DIR/mscore"
echo "You can now use: song2score score input.mp3 --pdf"
