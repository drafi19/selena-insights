"""Single source of truth for pipeline constants. Do not duplicate these numbers."""
from pathlib import Path

SEED = 42
TARGET_FPR = 0.01
ROLLING_WINDOW = 30
GAP_CLIP_DAYS = 60
FEATURES = [
    "log_amount",
    "dow",
    "dom",
    "month",
    "days_since_prev",
    "amount_z_30",
]
TIME_SPLIT = (0.70, 0.15, 0.15)
ARTIFACT_ROOT = Path("artifacts")
DEFAULT_VERSION = "v0.1.0"
INJECT_RATE = 0.10
EVAL_REPLICATES = 20
SYNTH_ROWS = 10_000
SYNTH_INJECT_RATE = 0.05
