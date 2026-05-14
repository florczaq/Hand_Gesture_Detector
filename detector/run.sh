#!/usr/bin/env bash
set -e

IMAGE="hand-gesture-detector-detector:latest"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Build the image if it doesn't exist yet (or pass --build to force rebuild).
if [[ "$1" == "--build" ]] || ! docker image inspect "$IMAGE" &>/dev/null; then
  echo "Building $IMAGE ..."
  docker build --network=host -t "$IMAGE" "$SCRIPT_DIR"
fi

# Allow local Docker containers to connect to the X server.
xhost +local:docker

cleanup() {
  xhost -local:docker
}
trap cleanup EXIT

docker run --rm -it \
  --device=/dev/video0:/dev/video0 \
  --group-add video \
  -e DISPLAY="$DISPLAY" \
  -e QT_QPA_PLATFORM=xcb \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$REPO_DIR":/workspace \
  -w /workspace \
  "$IMAGE" \
  python -m detector.detect

