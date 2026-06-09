$ErrorActionPreference = "Stop"

$IMAGE      = "hand-gesture-detector-detector:latest"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$REPO_DIR   = Split-Path -Parent $SCRIPT_DIR

# X11 display — requires VcXsrv running with "Disable access control" checked.
$DISPLAY = "host.docker.internal:0.0"

if ($args[0] -eq "--build" -or -not (docker image inspect $IMAGE 2>$null)) {
    Write-Host "Building $IMAGE ..."
    docker build --network=host -t $IMAGE $SCRIPT_DIR
}

docker run --rm -it `
    --device=/dev/video0:/dev/video0 `
    -e DISPLAY=$DISPLAY `
    -e QT_QPA_PLATFORM=xcb `
    -v "${REPO_DIR}:/workspace" `
    -w /workspace `
    $IMAGE `
    python -m detector.detect
