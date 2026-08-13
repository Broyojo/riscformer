#!/bin/bash
# Fetch the Doom pieces that aren't committed to this repo:
#   - doomgeneric source (GPL, cloned from upstream)
#   - doom1.wad (Doom 1.9 shareware, extracted from id's original installer)
# Requires: git, curl, 7z (brew install p7zip)
set -e
cd "$(dirname "$0")"

if [ ! -d doomgeneric ]; then
    echo "cloning doomgeneric..."
    git clone --depth 1 https://github.com/ozkl/doomgeneric.git doomgeneric
fi

if [ ! -f doom1.wad ]; then
    echo "fetching shareware doom1.wad (doom19s.zip from idgames mirror)..."
    tmp=$(mktemp -d)
    curl -sL -o "$tmp/doom19s.zip" "https://youfailit.net/pub/idgames/idstuff/doom/doom19s.zip"
    (cd "$tmp" && unzip -q doom19s.zip && cat DOOMS_19.1 DOOMS_19.2 > combined.sfx && 7z x -y combined.sfx > /dev/null)
    cp "$tmp/DOOM1.WAD" doom1.wad
    rm -rf "$tmp"
fi
ls -la doom1.wad
echo "ok"
