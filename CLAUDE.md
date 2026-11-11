# Standing rules for this project (read first, every session)

1. **Every script that is run gets saved to GitHub first.** Write or edit code in `dental_cls/` (or `tools/`, `benchmark/`), `git add` + commit + push to the
   working branch (`claude/eager-heisenberg-be64qy`) *before* running it on a server, and commit again after any fix. Nothing may exist only on Kaggle, in a
   notebook cell, or in a scratch directory. Inline one-off code that produced a reported number is saved as a script too.
2. **Results are backed up as they appear**: `python tools/sync_to_git.py --remote <server dir> --local backup/<dir> --every 900` (small files only), and on the
   server `tools/hf_upload_loop.py` pushes folders (weights, arrays) to Hugging Face so uploads continue without the session.
3. **Never commit secrets** (Hugging Face, Roboflow, GitHub tokens, Kaggle proxy URLs). Read them from environment variables. Before each commit run:
   `git grep -n -I -E 'hf_[A-Za-z0-9]{20,}|api_key *= *"[^"]+"|jupyter-proxy.kaggle.net/k/'` – it must print nothing.
4. **After each finished experiment**: add/refresh the `RESULTS_*.md`, update `HANDOFF.md` (what is done, what is next, how to re-run), then push.
5. Keep large files out of git; they go to Hugging Face `emad2001/dental-implant-brand-classification` (private). `/kaggle/working` has a 20 GB limit – use `/tmp/data`.
6. Fixed protocol: never regenerate `backup/crops/folds.json` / `benchmark/folds.json`; evaluation = group-stratified 5-fold, mean ± SD, paired tests with Holm correction.
7. Start a new session with: *"Continue the dental implant project. Read CLAUDE.md and HANDOFF.md on branch claude/eager-heisenberg-be64qy"* plus a Kaggle URL
   and a fresh Hugging Face token if GPU work or private files are needed.
