#!/usr/bin/env bash
# Push the serving image to a Hugging Face Space.
#   SPACE_URL="https://<user>:<token>@huggingface.co/spaces/<user>/selena-insights" #     bash scripts/sync_space.sh
set -euo pipefail
: "${SPACE_URL:?set SPACE_URL=https://user:TOKEN@huggingface.co/spaces/<user>/selena-insights}"
ARTIFACT="${ARTIFACT:-artifacts/v0.3.0}"

rm -rf build/space
mkdir -p build/space/artifacts
cp deploy/space/README.md deploy/space/Dockerfile requirements-serve.txt build/space/
cp -r selena_ml web build/space/
cp -r "$ARTIFACT" build/space/artifacts/

cd build/space
git init -q -b main
git add -A
git commit -qm "deploy $(date -u +%FT%TZ)"
git remote add space "$SPACE_URL" 2>/dev/null || git remote set-url space "$SPACE_URL"
git push -f space main
echo "deployed $ARTIFACT"
