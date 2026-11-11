#!/usr/bin/env python3
"""Copy local files to the Kaggle server:  python tools/kaggle_put.py /kaggle/working/dental_cls dental_cls/*.py
(base64 through the kernel, so it works without any other access; keep single files under ~20 MB)."""
import base64, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from kaggle_exec import run

dest, files = sys.argv[1], sys.argv[2:]
code = ["import base64, os", f"os.makedirs({dest!r}, exist_ok=True)"]
for f in files:
    code.append(f"open({dest + '/' + os.path.basename(f)!r}, 'wb').write(base64.b64decode({base64.b64encode(open(f, 'rb').read()).decode()!r}))")
code.append(f"print('uploaded', sorted(os.listdir({dest!r})))")
run("\n".join(code))
