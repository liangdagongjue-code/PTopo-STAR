"""Verify every distributed file against the public package manifest."""
from pathlib import Path
import csv,hashlib,json

root=Path(__file__).resolve().parents[1]
with (root/'FILE_MANIFEST.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
failures=[]
for row in rows:
    p=(root/row['path']).resolve()
    if not p.is_relative_to(root.resolve()) or not p.is_file(): failures.append(row['path']);continue
    if p.stat().st_size!=int(row['bytes']) or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
        failures.append(row['path'])
print(json.dumps({'checked_files':len(rows),'failed_files':failures}))
if failures:raise SystemExit(1)
