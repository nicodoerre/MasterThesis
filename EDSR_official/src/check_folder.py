# check_paths.py  – run:  python check_paths.py
from pathlib import Path
from data.diagrams import DIAGRAMS          # your custom class
from option import parser                   # repo’s argument parser

# --- set ONLY these two lines -------------------------------------------------
DIR_DATA = r"C:\Users\nicol\Desktop\UNI\3.Master\MasterThesis\PracticalPart\dataset_generation"
NAME     = "DIAGRAMS_MINI"                  # folder that holds train/ val/
# ------------------------------------------------------------------------------

# build a fake CLI just to instantiate the loader
cli = [
    "--data_train", NAME,
    "--data_test",  NAME,
    "--dir_data",   DIR_DATA,
    "--scale",      "2"
]
args = parser.parse_args(cli)

print("\nCreating DIAGRAMS train loader …")
ds = DIAGRAMS(args, train=False)

print("Built path:", ds.dir_hr)
hr_files = list(Path(ds.dir_hr).glob("*.*"))
print("HR files found:", len(hr_files))
if hr_files[:5]:
    print("First 3 files:", [p.name for p in hr_files[:3]])
