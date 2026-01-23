import os
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[2] / "output"
OUT_DIR.mkdir(exist_ok=True)

def save_frame(frame, name="frame.png"):
    path = OUT_DIR / name
    # placeholder: if frame is bytes write directly, else write repr
    mode = "wb" if isinstance(frame, (bytes, bytearray)) else "w"
    with open(path, mode) as f:
        if mode == "wb":
            f.write(frame)
        else:
            f.write(repr(frame))
    return str(path)

def save_metrics(metrics, name="metrics.txt"):
    p = OUT_DIR / name
    with open(p, "w") as f:
        f.write(str(metrics))
    return str(p)

__all__ = ["save_frame", "save_metrics"]
