# -*- coding: utf-8 -*-
"""02b_reshift.py — regenerate phenotypic-shift OOD images with the stronger
variant-C transform and re-extract features for val_shift / test_shift."""
import os
import sys

import numpy as np
import torch
import torchvision
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.set_num_threads(min(18, os.cpu_count() or 1))
CACHE = common.CACHE


def shift_image(img):
    from PIL import ImageEnhance, ImageFilter
    return (
        ImageEnhance.Brightness(
            ImageEnhance.Contrast(
                ImageEnhance.Color(img).enhance(0.50)
            ).enhance(1.45)
        ).enhance(1.30).filter(ImageFilter.GaussianBlur(1.5))
    )


class Subset(Dataset):
    def __init__(self, items, shift_dir):
        self.items = items
        self.shift_dir = shift_dir
        self.tf = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        p = os.path.join(self.shift_dir, it["name"].replace("/", "__"))
        with Image.open(p) as im:
            x = self.tf(im.convert("RGB"))
        return x, common.CLASS2IDX[it["cls"]]


def main():
    split = common.load_json("split.json")
    # 1) regenerate images
    for subset in ("val", "test"):
        out_dir = os.path.join(CACHE, "shift", subset)
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))
        for it in split[subset]:
            dst = os.path.join(out_dir, it["name"].replace("/", "__"))
            with Image.open(it["path"]) as im:
                shifted = shift_image(im.convert("RGB"))
            shifted.save(dst, quality=90)
        print(f"regenerated shift/{subset}: {len(os.listdir(out_dir))}", flush=True)

    # 2) re-extract features
    net = torchvision.models.resnet50(weights=None)
    net.load_state_dict(torch.load(common.WEIGHTS, map_location="cpu"))
    net.fc = torch.nn.Identity()
    net.eval()
    for subset in ("val", "test"):
        tag = f"{subset}_shift"
        loader = DataLoader(Subset(split[subset], os.path.join(CACHE, "shift", subset)),
                            batch_size=64, shuffle=False, num_workers=0)
        feats, labs = [], []
        with torch.no_grad():
            for x, y in loader:
                feats.append(net(x).numpy().astype(np.float32))
                labs.append(y.numpy().astype(np.int64))
        F = np.concatenate(feats, 0)
        Y = np.concatenate(labs, 0)
        np.save(os.path.join(CACHE, f"feat_{tag}.npy"), F)
        np.save(os.path.join(CACHE, f"lab_{tag}.npy"), Y)
        print(f"{tag}: {F.shape}", flush=True)
    print("RESHIFT DONE", flush=True)


if __name__ == "__main__":
    main()
