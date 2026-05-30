"""Resumable result IO + run manifest. Results are JSON; mpf -> float/str for serialisation."""
from __future__ import annotations
import json, os, time
import mpmath as mp

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))
FIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "figures"))
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logs"))
for d in (RESULTS_DIR, FIG_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)


def _coerce(o):
    if isinstance(o, mp.mpf):
        return float(o)
    if isinstance(o, mp.mpc):
        return [float(o.real), float(o.imag)]
    if isinstance(o, dict):
        return {k: _coerce(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_coerce(v) for v in o]
    return o


def save(name, obj):
    path = os.path.join(RESULTS_DIR, name if name.endswith(".json") else name + ".json")
    payload = {"_saved": time.strftime("%Y-%m-%d %H:%M:%S"), "data": _coerce(obj)}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    return path


def load(name):
    path = os.path.join(RESULTS_DIR, name if name.endswith(".json") else name + ".json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["data"]


def exists(name):
    path = os.path.join(RESULTS_DIR, name if name.endswith(".json") else name + ".json")
    return os.path.exists(path)


def log(msg, logfile="run.log"):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(LOG_DIR, logfile), "a") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    save("_io_selftest", {"a": mp.mpf("1.5"), "b": [mp.mpf(2), 3], "c": {"d": mp.mpf("0.1")}})
    print("roundtrip:", load("_io_selftest"))
    os.remove(os.path.join(RESULTS_DIR, "_io_selftest.json"))
    print("ok")
