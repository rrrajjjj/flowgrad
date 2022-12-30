import glob
from pathlib import Path


import librosa
import numpy as np
from PIL import Image
from torch.utils.data import IterableDataset
from ..utils import parse_annot


class UrbansasDataset(IterableDataset):
    def __init__(self, data_root, modal = "vision"):
        super(UrbansasDataset).__init__()
        self.data_root = data_root
        self.files = glob.glob("{}/Data/*.wav".format(self.data_root))
        self.urbansas_test = [Path(f).stem for f in self.files]
            
    def __iter__(self):
        for ft in self.urbansas_test:
            img = Image.open(
                [f for f in self.files if str(ft) in f][0].replace("wav", "jpg")
            ).convert("RGB")
            w, h = img.size

            audio, _ = librosa.load(
                [f for f in self.files if str(ft) in f][0]
            )

            bboxs = parse_annot("{}/Annotations/{}.txt".format(self.data_root,ft))
            gt_map = np.zeros([224, 224])
            
            for item in bboxs:
                x1, y1, bbox_w, bbox_h = int(item[0]/w*224), int(item[1]/h*224), int(item[2]/w*224), int(item[3]/h*224)
                x2, y2 = x1+bbox_w, y1+bbox_h
                temp = np.zeros([224, 224])
                temp[y1:y2, x1:x2] = 1
                gt_map += temp
            gt_map[gt_map > 1] = 1
            yield ft, img, audio, gt_map




