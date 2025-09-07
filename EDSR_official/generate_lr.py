import cv2, os
from pathlib import Path

ROOT   = Path(r"path/to/dataset")
SCALES = [2, 3, 4]

for split in ["train", "val", "test"]:
    split_dir = ROOT / split
    hr_dir    = split_dir / "HR"
    lr_root   = split_dir / "LR_bicubic"

    for hr_path in hr_dir.glob("*.*"):
        img = cv2.imread(str(hr_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Skipping unreadable file: {hr_path.name}")
            continue

        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        h, w = img.shape[:2]
        stem = hr_path.stem                            

        for s in SCALES:
            lr_dir = lr_root / f"X{s}"
            lr_dir.mkdir(parents=True, exist_ok=True)

            lr = cv2.resize(img, (w // s, h // s), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(lr_dir / f"{stem}x{s}.png"), lr)
