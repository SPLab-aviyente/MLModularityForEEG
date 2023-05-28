import sys

from pathlib import Path

PROJECT_DIR = str(Path(__file__).parents[1])

if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)