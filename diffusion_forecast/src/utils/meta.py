"""
Sidecar metadata I/O for the forecast pipeline.

Training (forecast_loader) writes a meta.json next to tst.npy recording the
stride/window/pred_len/n_basins actually used. Postprocessing reads meta.json
first so its settings cannot drift from the training run.
"""
import json
import os

META_FILENAME = 'meta.json'


def write_meta(pred_dir, **kwargs):
    """Write meta.json into pred_dir. Overwrites if exists."""
    os.makedirs(pred_dir, exist_ok=True)
    path = os.path.join(pred_dir, META_FILENAME)
    with open(path, 'w') as f:
        json.dump(kwargs, f, indent=2, default=str)
    return path


def read_meta(pred_dir):
    """Return dict from meta.json, or None if not present."""
    path = os.path.join(pred_dir, META_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
