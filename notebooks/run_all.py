"""
notebooks/run_all.py
Execute all 4 project notebooks top-to-bottom so the in-notebook output
cells match the current src/ code. Idempotent and safe to re-run.

Usage: python notebooks/run_all.py
       (or: from notebooks.run_all import _main; _main())
"""
import os
import sys

# Force UTF-8 stdout on Windows (cp1252 can't render ✓ / ✗ below)
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
NOTEBOOKS_DIR = Path(PROJECT_ROOT) / 'notebooks'

NOTEBOOKS = [
    '01_eda.ipynb',
    '02_preprocessing.ipynb',
    '03_clustering.ipynb',
    '04_business_insights.ipynb',
]


def _main():
    """Re-execute all 4 notebooks. Returns silently on missing deps so the
    pipeline can call this without a hard dependency on nbclient."""
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as e:
        print(f"  [notebooks] refresh skipped — {e}. "
              f"Install with: pip install nbformat nbclient")
        return

    for name in NOTEBOOKS:
        path = NOTEBOOKS_DIR / name
        print(f"\n=== {name} ===")
        if not path.exists():
            print(f"  not found: {path}")
            continue
        try:
            nb = nbformat.read(path, as_version=4)
            for cell in nb.cells:
                cell.pop('outputs', None)
                cell.pop('execution_count', None)
            client = NotebookClient(nb, timeout=600, kernel_name='python3',
                                    resources={'metadata': {'path': str(NOTEBOOKS_DIR)}})
            client.execute()
            nbformat.write(nb, path)
            print(f"  [OK] executed + saved")
        except Exception as e:
            print(f"  [FAIL] {e}")
            continue
    print("\nAll notebooks processed.")


if __name__ == '__main__':
    _main()
