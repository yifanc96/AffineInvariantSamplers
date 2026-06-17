#!/usr/bin/env bash
# Fetch the two Pantheon+SH0ES files needed for astro/run_pantheon.py.
# Source: github.com/PantheonPlusSH0ES/DataRelease (public, CC-BY).
set -euo pipefail

DEST="$(dirname "$0")/data"
mkdir -p "$DEST"

BASE="https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR"

curl -fL -o "$DEST/Pantheon+SH0ES.dat"             "$BASE/Pantheon%2BSH0ES.dat"
curl -fL -o "$DEST/Pantheon+SH0ES_STAT+SYS.cov"    "$BASE/Pantheon%2BSH0ES_STAT%2BSYS.cov"

echo "Downloaded to $DEST:"
ls -lh "$DEST"
