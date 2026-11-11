#!/usr/bin/env python3
"""Run ON the Kaggle server so uploads continue even if the Claude session or your computer is gone.

  HF_TOKEN=... nohup python hf_upload_loop.py --pairs /kaggle/working/results:results /tmp/data/det_runs:det_checkpoints --every 1200 &
Each pair is local_dir:path_in_repo. HF_TOKEN is read from the environment only (never written to disk or git)."""
import argparse, os, time
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="emad2001/dental-implant-brand-classification")
ap.add_argument("--pairs", nargs="+", required=True)
ap.add_argument("--every", type=int, default=1200)
ap.add_argument("--rounds", type=int, default=0, help="0 = forever")
a = ap.parse_args()
api = HfApi(token=os.environ["HF_TOKEN"])
k = 0
while True:
    for p in a.pairs:
        src, dst = p.split(":")
        try:
            api.upload_folder(folder_path=src, path_in_repo=dst, repo_id=a.repo, repo_type="dataset", ignore_patterns=["__pycache__", ".*"], commit_message=f"sync {dst}")
        except Exception as e:
            print("upload error", dst, e, flush=True)
    print(time.ctime(), "uploaded", a.pairs, flush=True)
    k += 1
    if a.rounds and k >= a.rounds:
        break
    time.sleep(a.every)
