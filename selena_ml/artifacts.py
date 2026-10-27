"""Save/load the artefact bundle. metadata.json is the single source of truth for numbers."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_metadata(artifact_dir: Path, metadata: dict) -> Path:
    payload = {
        **metadata,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path = Path(artifact_dir) / "metadata.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def read_metadata(artifact_dir: str | Path) -> dict:
    return json.loads((Path(artifact_dir) / "metadata.json").read_text())
