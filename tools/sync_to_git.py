#!/usr/bin/env python3
"""Pull result folders from the Kaggle server into this repo's backup/ and commit + push them.

  python tools/sync_to_git.py --remote /kaggle/working/results2 --local backup/results2 --msg "Backup round-2 results"
  python tools/sync_to_git.py ... --every 900          # repeat every 15 min until Ctrl-C / --stop-at N files
Only *.json, *.csv, *.log, *.png, *.md and *.txt are copied (no weights / arrays / secrets)."""
import argparse, base64, io, os, subprocess, sys, tarfile, time
sys.path.insert(0, os.path.dirname(__file__))
from kaggle_exec import run
import contextlib

ap = argparse.ArgumentParser()
ap.add_argument("--remote", required=True); ap.add_argument("--local", required=True)
ap.add_argument("--msg", default="Backup results"); ap.add_argument("--every", type=int, default=0)
ap.add_argument("--stop-at", type=int, default=0, help="stop once this many files exist locally")
a = ap.parse_args()
REMOTE = f"""
import base64, io, os, tarfile
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode='w:gz') as t:
    for root, _, fs in os.walk({a.remote!r}):
        for f in fs:
            if f.endswith(('.json', '.csv', '.log', '.png', '.md', '.txt')) and not f.startswith('.'):
                t.add(os.path.join(root, f), arcname=os.path.relpath(os.path.join(root, f), {a.remote!r}))
print('B64START' + base64.b64encode(buf.getvalue()).decode() + 'B64END')
"""


def once():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        run(REMOTE)
    s = out.getvalue()
    data = base64.b64decode(s.split("B64START")[1].split("B64END")[0])
    os.makedirs(a.local, exist_ok=True)
    tarfile.open(fileobj=io.BytesIO(data), mode="r:gz").extractall(a.local, filter="data")
    n = sum(len(f) for _, _, f in os.walk(a.local))
    subprocess.run(["git", "add", a.local], check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode:
        subprocess.run(["git", "commit", "-q", "-m", f"{a.msg} ({n} files)"], check=True)
        subprocess.run(["git", "push", "-q", "origin", "HEAD"], check=True)
    print(time.ctime(), "synced", n, "files", flush=True)
    return n


while True:
    n = once()
    if not a.every or (a.stop_at and n >= a.stop_at):
        break
    time.sleep(a.every)
