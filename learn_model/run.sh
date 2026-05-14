#!/usr/bin/env bash
set -e

IMAGE="hand-gesture-detector-learn-model:latest"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Build the image if it doesn't exist yet (or pass --build to force rebuild).
if [[ "$1" == "--build" ]] || ! docker image inspect "$IMAGE" &>/dev/null; then
  echo "Building $IMAGE ..."
  docker build -t "$IMAGE" "$SCRIPT_DIR"
fi

mkdir -p "$REPO_DIR/models"

docker run --rm -it \
  -v "$REPO_DIR":/workspace \
  -w /workspace \
  "$IMAGE" \
  python /workspace/learn_model/learn.py

