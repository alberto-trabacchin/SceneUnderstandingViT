from ultralytics import YOLO
import argparse
import torch
from pathlib import Path
import cv2 as cv
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import shutil
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--save-path', type=str, required=True)
    parser.add_argument('--resize', type=int, default=600, help='Resize images to this size (resize, resize)')
    parser.add_argument('--train-lb-size', type=int, default=None)
    return parser.parse_args()


def is_dangerous(frame_ann):
    dang_categories = {
        "pedestrian",
        # "car",
        # "truck",
        # "bus",
        # "motorcycle",
        # "bicycle"     
    }
    if "labels" not in frame_ann.keys():
        return False
    for label in frame_ann["labels"]:
        cat = label["category"]
        box = label["box2d"]
        box_height = box["x2"] - box["x1"]
        box_width = box["y2"] - box["y1"]
        if cat in dang_categories and box_height > 50 and box_width > 50:
            return True
        

def save_data(args, annotations, mode = "train"):
    print(f"Preparing {mode} data...")

    if mode == "test":
        Path(f"{args.save_path}/{mode}").mkdir(parents=True, exist_ok=True)
        pbar = tqdm(total=len(list(Path(f"{args.data_path}/images/100k/{mode}").rglob("*.jpg"))), position=0, leave=True)
        for img_path in Path(f"{args.data_path}/images/100k/{mode}").rglob("*.jpg"):
            img = cv.imread(str(img_path))
            img = cv.resize(img, (args.resize, args.resize))
            cv.imwrite(f"{args.save_path}/{mode}/{img_path.name}", img)
            pbar.update(1)
        pbar.close()
        return

    pbar = tqdm(total=len(annotations), position=0, leave=True)
    Path(f"{args.save_path}/{mode}/dangerous").mkdir(parents=True, exist_ok=True)
    Path(f"{args.save_path}/{mode}/safe").mkdir(parents=True, exist_ok=True)
    if mode == "train" and args.train_lb_size is not None:
        Path(f"{args.save_path}/unlabeled").mkdir(parents=True, exist_ok=True)
    safe_count = 0
    dangerous_count = 0
    for i, ann in enumerate(annotations):
        fname = ann["name"]
        img = cv.imread(f"{args.data_path}/images/100k/{mode}/{fname}")
        img = cv.resize(img, (args.resize, args.resize))
        if mode == "train" and args.train_lb_size is not None:
            if is_dangerous(ann) and (dangerous_count < args.train_lb_size//2):
                cv.imwrite(f"{args.save_path}/{mode}/dangerous/{fname}", img)
                dangerous_count += 1
            elif not is_dangerous(ann) and (safe_count < args.train_lb_size//2):
                cv.imwrite(f"{args.save_path}/{mode}/safe/{fname}", img)
                safe_count += 1
            else:
                cv.imwrite(f"{args.save_path}/unlabeled/{fname}", img)
        else:
            if is_dangerous(ann):
                cv.imwrite(f"{args.save_path}/{mode}/dangerous/{fname}", img)
            else:
                cv.imwrite(f"{args.save_path}/{mode}/safe/{fname}", img)
        pbar.update(1)
    pbar.close()


def get_labels(args, mode):
    with open(f"{args.data_path}/labels/det_20/det_{mode}.json", "r") as f:
        return json.load(f)



if __name__ == "__main__":
    args = parse_args()
    if Path(args.save_path).exists():
        shutil.rmtree(args.save_path)
    train_labels = get_labels(args, "train")
    val_labels = get_labels(args, "val")
    save_data(args, train_labels, "train")
    save_data(args, val_labels, "val")
    save_data(args, None, "test")