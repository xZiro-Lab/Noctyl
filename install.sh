#!/usr/bin/env bash
set -e
if pip install deep-scout; then
  echo "Installed deep-scout."
else
  echo "deep-scout package unavailable, falling back to legacy Noctyl source install."
  pip install "git+https://github.com/xZiro-Lab/Noctyl.git@main"
fi
