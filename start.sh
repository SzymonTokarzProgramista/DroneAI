#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  cat <<'EOF'
Error: uv is not installed.

Install uv first, then rerun:
  curl -LsSf https://astral.sh/uv/install.sh | sh

After installation, restart your shell or ensure that uv is on PATH.
EOF
  exit 1
fi

cd "$ROOT_DIR"

echo "[DroneAI] Creating or reusing local virtual environment..."
uv venv .venv

echo "[DroneAI] Syncing project dependencies..."
uv sync

echo "[DroneAI] Starting application..."
uv run python app.py
