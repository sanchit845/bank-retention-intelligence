"""Verify all `src.<mod>.<name>` references in notebooks still resolve."""
import json
import re
import sys
import os
import importlib

os.chdir(r'E:\bank-retention-intelligence')

def src_symbols_used(nb_path):
    with open(nb_path) as f:
        d = json.load(f)
    uses = []
    for cell in d['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        # `from src.MOD import NAME[, NAME, ...]`
        for m in re.finditer(r'from\s+src\.(\w+)\s+import\s+([^\n]+)', src):
            mod = m.group(1)
            for name in re.findall(r'(\w+)', m.group(2)):
                uses.append((mod, name))
    return uses

all_good = True
for nb in ['01_eda.ipynb', '02_preprocessing.ipynb', '03_clustering.ipynb', '04_business_insights.ipynb']:
    print(f'== {nb} ==')
    uses = src_symbols_used(f'notebooks/{nb}')
    by_mod = {}
    for m, n in uses:
        by_mod.setdefault(m, set()).add(n)
    for mod, names in sorted(by_mod.items()):
        try:
            module = importlib.import_module(f'src.{mod}')
            missing = [n for n in names if not hasattr(module, n)]
            if missing:
                print(f'  src.{mod}: MISSING {missing}')
                all_good = False
            else:
                print(f'  src.{mod}: OK ({len(names)} names)')
        except ImportError as e:
            print(f'  src.{mod}: IMPORT FAILED — {e}')
            all_good = False

sys.exit(0 if all_good else 1)
