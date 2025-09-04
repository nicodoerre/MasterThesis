# src/data/diagrams.py
import os
from data.srdata import SRData 

class DIAGRAMS(SRData):
    """
    Custom dataset class for own data
    Dataset layout expected on disk
    ───────────────────────────────
    <dir_data>/DIAGRAMS/
        train/HR/              *.png  (RGB, HR originals)
        train/LR_bicubic/X2/   *.png
        train/LR_bicubic/X3/   *.png
        train/LR_bicubic/X4/   *.png
        val/HR/                …
        val/LR_bicubic/X2/…X4/ …
    """

    def __init__(self, args, name='DIAGRAMS', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        self.begin, self.end = map(int, data_range[0 if train else 1])

        super(DIAGRAMS, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, dir_data):
        split = 'train' if self.train else 'test'
        self.apath  = os.path.join(dir_data, 'DIAGRAMS')        # dataset root
        self.dir_hr = os.path.join(self.apath, split, 'HR')
        self.dir_lr = os.path.join(self.apath, split, 'LR_bicubic')
        self.ext = ('.png', '.png')
