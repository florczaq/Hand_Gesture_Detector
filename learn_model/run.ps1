$ErrorActionPreference = "Stop"

$IMAGE     = "hand-gesture-detector-learn-model:latest"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$REPO_DIR   = Split-Path -Parent $SCRIPT_DIR

if ($args[0] -eq "--build" -or -not (docker image inspect $IMAGE 2>$null)) {
    Write-Host "Building $IMAGE ..."
    docker build -t $IMAGE $SCRIPT_DIR
}

New-Item -ItemType Directory -Force -Path "$REPO_DIR\models" | Out-Null

docker run --rm -it `
    -v "${REPO_DIR}:/workspace" `
    -w /workspace `
    $IMAGE `
    python /workspace/learn_model/learn.py
