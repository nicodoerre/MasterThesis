# convert_hr_to_rgb.py
from pathlib import Path
from PIL import Image

ROOT = Path(r"C:/Users/nicol/Desktop/UNI/3.Master/MasterThesis/PracticalPart/dataset_generation")

for split in ["train", "valid", "test"]:
    hr_dir = ROOT / split / "HR"
    for p in hr_dir.glob("*.*"):              # every file in HR/
        img = Image.open(p).convert("RGB")    # strips alpha if present
        img.save(p)                           # overwrites in-place
        print("✔︎", p.name)
