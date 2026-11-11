#!/usr/bin/env python3
"""Run code on a Kaggle Jupyter server from anywhere (needs `pip install websocket-client`).

  export KAGGLE_URL='https://kkb-production.jupyter-proxy.kaggle.net/k/<id>/<token>/proxy'   # never commit this
  python tools/kaggle_exec.py 'print(1+1)'          # inline code
  python tools/kaggle_exec.py path/to/script.py     # or a file
Uses the first running kernel (override with KAGGLE_KERNEL=<id>). Prints stdout, results and tracebacks."""
import json, os, sys, urllib.request, uuid
import websocket


def kernel_id(base):
    if os.environ.get("KAGGLE_KERNEL"):
        return os.environ["KAGGLE_KERNEL"]
    ks = json.load(urllib.request.urlopen(base + "/api/kernels", timeout=30))
    if not ks:
        sys.exit("no running kernel")
    return ks[0]["id"]


def run(code, base=None, timeout=3600):
    base = base or os.environ["KAGGLE_URL"].rstrip("/")
    ws = websocket.create_connection(base.replace("https", "wss", 1) + f"/api/kernels/{kernel_id(base)}/channels", timeout=timeout)
    mid = uuid.uuid4().hex
    ws.send(json.dumps({"header": {"msg_id": mid, "username": "u", "session": uuid.uuid4().hex, "msg_type": "execute_request", "version": "5.3"},
                        "parent_header": {}, "metadata": {}, "content": {"code": code, "silent": False}, "channel": "shell"}))
    while True:
        m = json.loads(ws.recv())
        if m["parent_header"].get("msg_id") != mid:
            continue
        t, c = m["msg_type"], m["content"]
        if t == "stream":
            print(c["text"], end="")
        elif t in ("execute_result", "display_data"):
            print(c["data"].get("text/plain"))
        elif t == "error":
            print("\n".join(c["traceback"]))
        elif t == "status" and c["execution_state"] == "idle":
            break


if __name__ == "__main__":
    a = sys.argv[1]
    run(open(a).read() if os.path.isfile(a) else a)
