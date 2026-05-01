#!/usr/bin/env bash
# Download + extract the DareFightingICE 7.1 macOS-ready release.
#
# This is the path for monitoring-on-Mac: the JVM runs natively (LWJGL has
# Apple Silicon natives bundled in the release), so the game window opens
# directly on the Mac with full sprites — no Xvfb, no VNC, no streaming.
#
# Prereq: Java 21
#   brew install openjdk@21
#   echo 'export PATH="/opt/homebrew/opt/openjdk@21/bin:$PATH"' >> ~/.zshrc
#
# Usage: ./scripts/fetch_native.sh   (or `make fetch-native`)
set -euo pipefail

VERSION="7.1"
ZIP_URL="https://github.com/TeamFightingICE/FightingICE/releases/download/v${VERSION}/DareFightingICE-${VERSION}.zip"
RESOURCE_URL="https://github.com/TeamFightingICE/FightingICE/releases/download/v${VERSION}/resource-${VERSION}.zip"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR_DIR="${REPO_ROOT}/vendor/fightingice"
TMP_ZIP="/tmp/DareFightingICE-${VERSION}.zip"
TMP_RESOURCE_ZIP="/tmp/resource-${VERSION}.zip"

if [ -d "$VENDOR_DIR" ] && [ -f "$VENDOR_DIR/FightingICE.jar" ] && [ -d "$VENDOR_DIR/data" ]; then
    echo "[fetch_native] already installed at $VENDOR_DIR"
    exit 0
fi

mkdir -p "$VENDOR_DIR"

# Download and extract the game engine
if [ ! -f "$VENDOR_DIR/FightingICE.jar" ]; then
    echo "[fetch_native] downloading $ZIP_URL"
    curl -fL -o "$TMP_ZIP" "$ZIP_URL"
    echo "[fetch_native] engine zip: $(du -h "$TMP_ZIP" | cut -f1)"

    unzip -q -o "$TMP_ZIP" -d "$VENDOR_DIR"

    # The zip may extract to a subdirectory like DareFightingICE-7.1/. Flatten it.
    inner=$(find "$VENDOR_DIR" -maxdepth 2 -name FightingICE.jar -type f | head -1)
    if [ -n "$inner" ] && [ "$(dirname "$inner")" != "$VENDOR_DIR" ]; then
        inner_dir="$(dirname "$inner")"
        echo "[fetch_native] flattening $inner_dir into $VENDOR_DIR"
        cp -R "$inner_dir"/* "$VENDOR_DIR/"
        rm -rf "$inner_dir"
    fi

    rm -f "$TMP_ZIP"
fi

# Download and extract the resource pack (characters, graphics, sounds)
if [ ! -d "$VENDOR_DIR/data" ]; then
    echo "[fetch_native] downloading $RESOURCE_URL"
    curl -fL -o "$TMP_RESOURCE_ZIP" "$RESOURCE_URL"
    echo "[fetch_native] resource zip: $(du -h "$TMP_RESOURCE_ZIP" | cut -f1)"

    unzip -q -o "$TMP_RESOURCE_ZIP" -d "$VENDOR_DIR"

    # Flatten if extracted to a subdirectory
    if [ ! -d "$VENDOR_DIR/data" ]; then
        inner_data=$(find "$VENDOR_DIR" -maxdepth 2 -type d -name "data" | head -1)
        if [ -n "$inner_data" ]; then
            inner_dir="$(dirname "$inner_data")"
            echo "[fetch_native] flattening resources from $inner_dir"
            cp -R "$inner_dir"/* "$VENDOR_DIR/"
            rm -rf "$inner_dir"
        fi
    fi

    rm -f "$TMP_RESOURCE_ZIP"
fi

# Make the macOS run script executable.
chmod +x "$VENDOR_DIR"/*.sh 2>/dev/null || true

echo "[fetch_native] installed to $VENDOR_DIR"
ls "$VENDOR_DIR" | head -10
