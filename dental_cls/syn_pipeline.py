"""Server-side, self-running pipeline for the synthetic-Bego experiment (survives a dropped client connection; every stage saved to Hugging Face).
Stages: restore data -> train missing fold generators (2 GPUs) -> sample -> QC -> build crops_syn -> 30 classifier jobs -> analysis.
Run: setsid nohup python syn_pipeline.py > /kaggle/working/syn_pipeline.log 2>&1 &   (needs /root/.cache/huggingface/token)"""
import glob, os, shutil, subprocess, time
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
REPO = "emad2001/dental-implant-brand-classification"; api = HfApi(); W = "/kaggle/working"; CODE = f"{W}/repo/dental_cls"; O = f"{W}/cdiff"
D, DS = "/tmp/data/crops", "/tmp/data/crops_syn"
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def sh(c): log("$", c[:160]); r = subprocess.run(c, shell=True, capture_output=True, text=True, cwd=CODE); log((r.stdout + r.stderr)[-1500:]); return r.returncode
def up(folder, path, patterns=None, msg=None):
    for i in range(3):
        try: api.upload_folder(folder_path=folder, path_in_repo=path, repo_id=REPO, repo_type="dataset", allow_patterns=patterns, commit_message=msg or path); return
        except Exception as e: log("upload failed", e); time.sleep(30)

# 1. restore
os.makedirs(D, exist_ok=True); os.makedirs(O, exist_ok=True)
if not os.path.exists(f"{D}/raw_orig.npy"):
    shutil.copy(hf_hub_download(REPO, "crops/raw_orig.npy", repo_type="dataset"), f"{D}/raw_orig.npy")
for f in ("meta.csv", "folds.json"): shutil.copy(f"{W}/repo/backup/crops/{f}", f"{D}/{f}")
p = snapshot_download(REPO, repo_type="dataset", allow_patterns=["cdiff/*"])
for f in glob.glob(f"{p}/cdiff/*"): shutil.copy(f, O)
for f in ("r0_f0_raw384_densenet121.pt", "r0_f1_raw384_densenet121.pt", "r0_f2_raw384_densenet121.pt", "r0_f3_raw384_densenet121.pt", "r0_f4_raw384_densenet121.pt"):
    hf_hub_download(REPO, f"checkpoints_ts/{f}", repo_type="dataset", local_dir="/tmp/data")
log("restored; generators present:", sorted(os.path.basename(f) for f in glob.glob(f"{O}/cdiff_f*.pt")))

# 2. generators for the missing folds, 2 GPUs in parallel
todo = [k for k in range(5) if not os.path.exists(f"{O}/syn_f{k}.npy")]
queues = {0: todo[0::2], 1: todo[1::2]}; procs = []
for gpu, folds in queues.items():
    if not folds: continue
    body = " && ".join(
        (f"( test -f {O}/cdiff_f{k}.pt || python cdiff_brand.py train --fold {k} --crops {D} --out {O} --epochs 120 ) && "
         f"python cdiff_brand.py sample --fold {k} --crops {D} --out {O} --n_bego 400 --n_bicon 32 --n_iti 32 && "
         f"python -c \"from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='{O}', path_in_repo='cdiff', repo_id='{REPO}', repo_type='dataset', allow_patterns=['*f{k}*'], commit_message='cdiff fold {k}')\"")
        for k in folds)
    open(f"{W}/gen{gpu}.sh", "w").write(f"cd {CODE}\nexport CUDA_VISIBLE_DEVICES={gpu}\n{body}\n")
    procs.append(subprocess.Popen(f"bash {W}/gen{gpu}.sh > {W}/gen{gpu}.log 2>&1", shell=True))
log("generator queues", queues)
for pr in procs: pr.wait()
missing = [k for k in range(5) if not os.path.exists(f"{O}/syn_f{k}.npy")]
if missing: log("STOP: synthetic sets missing for folds", missing); raise SystemExit(1)

# 3. quality check, 4. training set
sh(f"python qc_syn.py --syn {O} --crops {D} --ckpt /tmp/data/checkpoints_ts --code . --out {W}/report_syn_qc")
up(f"{W}/report_syn_qc", "report_syn_qc")
sh(f"python build_syn.py {D} {O} {DS} 0")

# 5. classifiers: DenseNet121 + ViT-B/16, 3 seeds, synthetic Bego in training only
T = "python train_cv.py --crops " + DS + " --seed {s} --out " + W + "/results_syn/s{s} --versions raw_orig --models {m}"
q = {0: [T.format(s=s, m="densenet121") for s in (0, 1, 2)] + [T.format(s=2, m="vit_b_16")], 1: [T.format(s=s, m="vit_b_16") for s in (0, 1)]}
procs = []
for g, cmds in q.items():
    open(f"{W}/c{g}.sh", "w").write(f"cd {CODE}\nexport CUDA_VISIBLE_DEVICES={g}\n" + "\n".join(cmds) + "\n")
    procs.append(subprocess.Popen(f"bash {W}/c{g}.sh > {W}/c{g}.log 2>&1", shell=True))
while any(pr.poll() is None for pr in procs):
    time.sleep(600); up(f"{W}/results_syn", "results_syn", ["s*/*.json"], "results_syn (partial)")
up(f"{W}/results_syn", "results_syn", ["s*/*.json"], "results_syn")

# 6. analysis
sh(f"cd {W} && python {W}/k_an_syn.py > {W}/an_syn.txt 2>&1; tail -40 {W}/an_syn.txt")
up(f"{W}/report_syn", "report_syn"); api.upload_file(path_or_fileobj=f"{W}/an_syn.txt", path_in_repo="report_syn/analysis_output.txt", repo_id=REPO, repo_type="dataset")
log("PIPELINE DONE")
