# -*- coding: utf-8 -*-
"""02_extract_features.py
Extract 2048-d features with a frozen ImageNet-pretrained ResNet-50 for all
subsets (train/val/test and their phenotypic-shift OOD variants). Features are
cached as float32 npy files; downstream training is fast on CPU.
"""
import os
import sys

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

torch.set_num_threads(min(18, os.cpu_count() or 1))

CACHE = common.CACHE
os.makedirs(CACHE, exist_ok=True)


class Subset(Dataset):
    def __init__(self, items, shift_dir=None):
        self.items = items
        self.shift_dir = shift_dir
        self.tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(common.IMAGENET_MEAN, common.IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        if self.shift_dir is not None:
            p = os.path.join(self.shift_dir, it["name"].replace("/", "__"))
        else:
            p = it["path"]
        with Image.open(p) as im:
            x = self.tf(im.convert("RGB"))
        return x, common.CLASS2IDX[it["cls"]]


def extract(net, loader, tag):
    feats, labs = [], []
    with torch.no_grad():
        for k, (x, y) in enumerate(loader):
            z = net(x)
            feats.append(z.numpy().astype(np.float32))
            labs.append(y.numpy().astype(np.int64))
            if k % 20 == 0:
                print(f"  {tag}: batch {k}/{len(loader)}", flush=True)
    F = np.concatenate(feats, 0)
    Y = np.concatenate(labs, 0)
    np.save(os.path.join(CACHE, f"feat_{tag}.npy"), F)
    np.save(os.path.join(CACHE, f"lab_{tag}.npy"), Y)
    print(f"{tag}: {F.shape}", flush=True)
    return F, Y


def main():
    split = common.load_json("split.json")
    net = torchvision.models.resnet50(weights=None)
    sd = torch.load(common.WEIGHTS, map_location="cpu")
    net.load_state_dict(sd)
    net.fc = torch.nn.Identity()
    net.eval()
    print("backbone ready (frozen ResNet-50)", flush=True)

    loader = lambda items, shift: DataLoader(  # noqa: E731
        Subset(items, shift), batch_size=64, shuffle=False, num_workers=0)

    extract(net, loader(split["train"], None), "train")
    extract(net, loader(split["val"], None), "val")
    extract(net, loader(split["test"], None), "test")
    extract(net, loader(split["val"], os.path.join(CACHE, "shift", "val")), "val_shift")
    extract(net, loader(split["test"], os.path.join(CACHE, "shift", "test")), "test_shift")


if __name__ == "__main__":
    main()
