from ultralytics import YOLO
import argparse
import torch
from pathlib import Path
import cv2 as cv
from tqdm import tqdm
from matplotlib import pyplot as plt
import os




def setup():
    os.environ["YOLO_VERBOSE"] = "False"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--save-path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resize', type=int, default=600, help='Resize images to this size (resize, resize)')
    parser.add_argument('--conf-threshold', type=float, default=0.7)
    parser.add_argument('--train-lb-size', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=110)
    parser.add_argument('--downsample', type=int, default=10, help='Downsample the video by this factor')
    return parser.parse_args()


def get_frames(args, vid_path):
    vid = cv.VideoCapture(str(vid_path))
    frames = []
    count = 0
    while vid.isOpened():
        ret, frame = vid.read()
        count += 1
        if not ret:
            break
        if (count % args.downsample) == 0:
            frames.append(frame)
    return frames


def split_frames_batch(frames, batch_size):
    for i in range(0, len(frames), batch_size):
        yield frames[i : i + batch_size]
    return frames


def is_dangerous(args, pred):
    if pred == -1:
        return False
    for b, p in zip(pred.boxes.xyxy, pred.boxes.conf):
        if p > args.conf_threshold:
            return True
    return False


def save_vid_frames(args, vid_frames, targets, vid_name):
    vid_save_path = Path(f"{args.save_path}")
    Path(vid_save_path / "safe").mkdir(parents=True, exist_ok=True)
    Path(vid_save_path / "dangerous").mkdir(parents=True, exist_ok=True)
    for i, (f, t) in enumerate(zip(vid_frames, targets)):
        f = cv.resize(f, (args.resize, args.resize))
        if t == 0:
            cv.imwrite(f"{args.save_path}/safe/{vid_name}_{i}.jpg", f)
        else:
            cv.imwrite(f"{args.save_path}/dangerous/{vid_name}_{i}.jpg", f)


def classify_images(args, mode = 'train'):
    vid_paths = [p for p in Path(f"{args.data_path}/videos/{mode}").iterdir() if p.is_file()]
    model = YOLO("yolov8x.pt", verbose=False)
    print(f'Processing {mode} data')
    pbar = tqdm(total=len(vid_paths))
    safe_counter = 0
    dang_counter = 0
    dang_classes = {
        "person": 0,
        "bicycle": 1,
        "car": 2,
        "motorcycle": 3,
        "bus": 5, 
        "truck": 7
    }
    for p in vid_paths:
        targets = []
        vid_frames = get_frames(args, p)
        batch_frames = list(split_frames_batch(vid_frames, args.batch_size))
        vid_preds = []
        for b in batch_frames:
            preds = model.predict(b, classes = list(dang_classes.values()))
            if len(preds) == 0:
                vid_preds.extend(-1)
            else:
                vid_preds.extend(preds)
        for pred in vid_preds:
            if is_dangerous(args, pred):
                targets.append(1)
            else:
                targets.append(0)
        save_vid_frames(args, vid_frames, targets, p.stem)
        safe_counter += targets.count(0)
        dang_counter += targets.count(1)
        pbar.update(1)
        pbar.set_description(
            f"S: {safe_counter} | D: {dang_counter} | " \
            f"SR: {(safe_counter / (safe_counter + dang_counter)):.2f} | " \
            f"DR: {(dang_counter / (safe_counter + dang_counter)):.2f}"
        )
    pbar.close()
    
    


if __name__ == "__main__":
    args = parse_args()
    setup()
    classify_images(args, mode = 'train')