"""
main.py
Bank Retention Intelligence Platform — Master Runner

Runs the full analytics pipeline end-to-end via src.pipeline.run_full_pipeline.

Usage:
    python main.py
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# Force UTF-8 stdout/stderr on Windows so box-drawing characters in banners
# don't crash on cp1252. No-op on macOS/Linux (utf-8 by default).
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')

from src.pipeline import run_full_pipeline
from src.utils    import print_section


def main():
    print_section("Bank Retention Intelligence Platform — Full Pipeline")
    t0 = time.time()
    run_full_pipeline()
    print(f"\n  Total wall time: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
