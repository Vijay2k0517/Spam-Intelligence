"""CLI runner script for training models from project root or scripts directory.

Usage:
    python scripts/train_models.py --data data/raw/SMSSpamCollection --output-dir artifacts --seed 42
"""

from pathlib import Path
import sys

# Add project root and backend directory to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

for d in [str(BACKEND_DIR), str(PROJECT_ROOT)]:
    if d not in sys.path:
        sys.path.insert(0, d)

from app.ml.train import main

if __name__ == "__main__":
    main()
