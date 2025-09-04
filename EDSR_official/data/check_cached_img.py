# check_cache_channels.py  ← run inside the repo's venv
from pathlib import Path
import pickle, torch

ROOT = Path(r"C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/PracticalPart/dataset_generation")

bad = []
for p in ROOT.rglob("*.pt"):
    with open(p, "rb") as f:
        t = pickle.load(f)        # numpy array
        if t.ndim == 3 and t.shape[2] != 3:
            bad.append(p)
            if len(bad) == 5: break

print("Found", len(bad), "cached tensors with", 
      "channels =", {pickle.load(open(x, 'rb')).shape[2] for x in bad})
