#!/usr/bin/env bash
# WANDS is MIT licensed. 42,994 products / 480 queries / 233,448 judgments.
set -euo pipefail
mkdir -p data
cd data
if [ ! -d WANDS ]; then
  git clone --depth 1 https://github.com/wayfair/WANDS.git
fi
ls -la WANDS/dataset