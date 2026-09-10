#!/usr/bin/env python3
"""Exact frozen-value lookup. No calculation, data imports, or network."""
import argparse,csv,json
from pathlib import Path
B=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--pointer');p.add_argument('--search');a=p.parse_args()
if a.pointer:
 d=json.loads((B/'sources/results/paper_validation_snapshot.json').read_text())
 for k in a.pointer.strip('/').split('/'):
  k=k.replace('~1','/').replace('~0','~');d=d[int(k)] if isinstance(d,list) else d[k]
 print(json.dumps(d,ensure_ascii=False,indent=2))
elif a.search:
 with (B/'numeric_index.csv').open() as f:
  for row in csv.DictReader(f):
   if a.search in row['snapshot_pointer']:print(json.dumps(row,ensure_ascii=False))
else:p.error('Use --pointer /evidence_rows/1 or --search brier')
