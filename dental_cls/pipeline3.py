"""Self-running pipeline: (1a) Lashaki protocol, (1b) Tekis protocol, (2) open-set rejection. Each stage is saved to Hugging Face.
Run: setsid nohup python pipeline3.py > /kaggle/working/p3.log 2>&1 &   (needs the HF token in /root/.cache/huggingface/token)"""
import os, subprocess, threading, time
from huggingface_hub import HfApi
W, CODE = "/kaggle/working", "/kaggle/working/repo/dental_cls"; REPO = "emad2001/dental-implant-brand-classification"; api = HfApi()
ENV = dict(os.environ, ROBOFLOW_API_KEY="c0IWgB7XKPnqth9eD8Qy")
LA, TK, OS = "/tmp/data/lashaki", "/tmp/data/tekis", f"{W}/openset"
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def sh(c, gpu=None, logf=None):
    env = dict(ENV, **({"CUDA_VISIBLE_DEVICES": str(gpu)} if gpu is not None else {}))
    log("$", c[:170]); r = subprocess.run(c, shell=True, capture_output=True, text=True, cwd=CODE, env=env)
    txt = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "arning" not in l and "it/s]" not in l)
    if logf: open(logf, "a").write(txt + "\n")
    log(txt[-1200:]); return r.returncode
def up(folder, path, patterns=None):
    for i in range(3):
        try: api.upload_folder(folder_path=folder, path_in_repo=path, repo_id=REPO, repo_type="dataset", allow_patterns=patterns, commit_message=path); return
        except Exception as e: log("upload failed", e); time.sleep(30)
def wait(path):
    while not os.path.exists(path): time.sleep(20)

# setup: packages, then downloads (background) and data preparation (CPU)
sh("pip install -q ultralytics roboflow 2>&1 | tail -1")
dl = ("mkdir -p /tmp/data/oii /tmp/data/zen && cd /tmp/data/oii && (test -d OII-DS || (aria2c -q -x 16 -s 16 -o oii.rar https://ndownloader.figshare.com/files/40131892 "
      "&& unrar x -o+ -inul oii.rar && rm oii.rar)) && cd /tmp/data/zen && (test -d train || (aria2c -q -x 16 -s 16 -o zen.zip "
      "'https://zenodo.org/api/records/15487430/files/panoramic_radiography_yolo_dataset_14_classes.zip/content' && unzip -q -o zen.zip && rm zen.zip)) "
      f"&& cd {CODE} && python crops_from_yolo.py /tmp/data/zen /tmp/data/zen_crops 0 && touch /tmp/data/ood_ready")
threading.Thread(target=sh, args=(dl,), daemon=True).start()
sh(f"python tekis_yolo.py prep --out {TK}"); up(TK, "protocols/tekis", ["images.csv"])
threading.Thread(target=lambda: (sh(f"python lashaki_protocol.py prep --out {LA}"), open("/tmp/data/lashaki_ready", "w").close()), daemon=True).start()

jobs = [f"{t}:{p}:{k}" for t in ("bicon_bego", "3class") for p in ("leaky", "grouped", "grouped_nomask", "grouped_raw") for k in range(5)]
def gpu0():
    sh(f"python tekis_yolo.py train --out {TK} --split rf", 0, f"{W}/tekis_rf.log"); up(TK, "protocols/tekis", ["result_*.json", "images.csv"])
    wait("/tmp/data/lashaki_ready")
    sh(f"python lashaki_protocol.py run --out {LA} --jobs {' '.join(jobs[:24])}", 0, f"{W}/lashaki0.log"); up(LA, "protocols/lashaki", ["results/*.json", "images.csv"])
def gpu1():
    sh(f"python tekis_yolo.py train --out {TK} --split grouped", 1, f"{W}/tekis_grouped.log"); up(TK, "protocols/tekis", ["result_*.json", "images.csv"])
    for m in ("densenet121", "vit_b_16"):
        sh(f"python train_cv.py --crops /tmp/data/crops --seed 0 --out {W}/results_os --versions raw_orig --models {m} --save_all", 1, f"{W}/os_train.log")
    wait("/tmp/data/ood_ready")
    sh(f"python openset.py --ckpt {W}/results_os_ckpt --crops /tmp/data/crops --code . --out {OS} "
       f"--ood oii=/tmp/data/oii/OII-DS/Oral-Implant-image/single_implants zenodo=/tmp/data/zen_crops", 1, f"{W}/openset.log")
    up(OS, "protocols/openset"); up(f"{W}/results_os", "protocols/openset/results_fold_models", ["*.json"])
    wait("/tmp/data/lashaki_ready")
    sh(f"python lashaki_protocol.py run --out {LA} --jobs {' '.join(jobs[24:])}", 1, f"{W}/lashaki1.log"); up(LA, "protocols/lashaki", ["results/*.json", "images.csv"])
t = [threading.Thread(target=f) for f in (gpu0, gpu1)]
for x in t: x.start()
for x in t: x.join()
log("PIPELINE3 DONE")
