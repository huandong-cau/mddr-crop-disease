# -*- coding: utf-8 -*-
"""Shared constants and utilities for the MDDR benchmark pipeline."""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "cache")
WEIGHTS = r"C:\Users\HUAWEI\Documents\deepseek-harness\default-workspace\journal_transfer\weights\resnet50-0676ba61.pth"
RESULTS = os.path.join(os.path.dirname(BASE), "results")
DATA_ROOT = r"C:\Users\HUAWEI\Downloads\Plant_leaf_diseases_dataset_with_augmentation\Plant_leave_diseases_dataset_with_augmentation"

CLASS_FOLDERS = {
    "gls": "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "healthy": "Corn___healthy",
    "nclb": "Corn___Northern_Leaf_Blight",
    "rust": "Corn___Common_rust",
}
CLASS_NAMES = ["gls", "healthy", "nclb", "rust"]  # canonical order
CLASS2IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

N_SEEDS = 5
SPLIT_SEED = 42
ENSEMBLE_SIZE = 5
MC_PASSES = 30

# agronomic ontology for maize (deterministic, documented rules used by the
# ontological-doubt module; growth stage / weather are simulated deployment metadata)
ONTOLOGY_RULES = {
    # GLS lesion maturation requires >= V6; at V4/V5 a GLS spray recommendation is inconsistent
    "gls": {"min_stage": "V6", "stage_order": ["V4", "V5", "V6", "V7", "VT", "R1"]},
    # common rust favoured by 16-25 C; above 30 C rust is agronomically unlikely
    "rust": {"max_temp_c": 28.0},
}


def load_json(name):
    with open(os.path.join(CACHE, name), "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, name):
    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
