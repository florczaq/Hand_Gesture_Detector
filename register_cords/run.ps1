$ErrorActionPreference = "Stop"

$IMAGE      = "hand-gesture-detector:latest"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

# X11 display — requires VcXsrv running with "Disable access control" checked.
$DISPLAY = "host.docker.internal:0.0"

if ($args[0] -eq "--build" -or -not (docker image inspect $IMAGE 2>$null)) {
    Write-Host "Building $IMAGE ..."
    docker build -t $IMAGE $SCRIPT_DIR
}

docker run --rm -it `
    --device=/dev/video0:/dev/video0 `
    -e DISPLAY=$DISPLAY `
    -e QT_QPA_PLATFORM=xcb `
    -v "${SCRIPT_DIR}:/app" `
    -w /app `
    $IMAGE
