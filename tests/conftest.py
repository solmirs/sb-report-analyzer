"""Pytest configuration — ensures project root is on sys.path so that
``import config``, ``import models`` etc. work from the tests/ directory.
"""

import sys
from pathlib import Path

# Add project root (parent of tests/) to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
