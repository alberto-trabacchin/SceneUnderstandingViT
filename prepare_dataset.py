from pathlib import Path
import argparse
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
import torch
from PIL import Image
from torchvision.transforms import transforms
import tqdm
import shutil
import collections
import random


parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='./data')
parser.add_argument('--save-path', type=str, default='./dreyeve')
parser.add_argument('--train-lb-size', type=int, default=10000)
parser.add_argument('--val-size', type=int, default=1000)
parser.add_argument('--test-size', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--resize', type=int, nargs='+', default=[216, 384], help='Resize images to this size (width, height)')
parser.add_argument('--conf-threshold', type=float, default=0.7)

args = parser.parse_args()
args.device = torch.device(args.device)


def split_data(args):
    data_path = Path(args.data_path)
    images_paths = [str(p) for p in data_path.iterdir() if p.suffix == '.jpg']
    random.shuffle(images_paths)
    train_lb_size = args.train_lb_size
    val_size = args.val_size
    test_size = args.test_size
    train_ul_size = len(images_paths) - val_size - test_size
    train_lb_data = images_paths[:train_lb_size]
    train_ul_data = images_paths[:train_ul_size]
    val_data = images_paths[train_ul_size:train_ul_size + val_size]
    test_data = images_paths[train_ul_size + val_size:]
    assert(len(train_lb_data) == train_lb_size)
    assert(len(train_ul_data) == train_ul_size)
    assert(len(val_data) == val_size)
    assert(len(test_data) == test_size)

    return train_lb_data, train_ul_data, val_data, test_data


def detect_targets(args, data_paths, mode):
    model = retinanet_resnet50_fpn_v2(weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
    model = model.to(args.device)
    model.eval()
    if args.resize is not None:
        img_transform = transforms.Compose([
            transforms.Resize(size=tuple(args.resize), antialias=True),
            transforms.ToTensor()
        ])
    else:
        img_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    outputs = []
    pbar = tqdm.tqdm(total=len(data_paths), desc=f'Processing {mode} data')
  
    for i in range(0, len(data_paths), args.batch_size):
        paths_batch = data_paths[i:i+args.batch_size]
        data_batch = []

        for p in paths_batch:
            img = Image.open(p)
            img = img_transform(img)
            data_batch.append(img)
        data_batch = torch.stack(data_batch).to(args.device)
        with torch.inference_mode():
            preds = model(data_batch)
            outputs.extend(preds)
        pbar.update(len(paths_batch))
    
    del model
    return outputs


def save_data(args, data_paths, targets, mode):
    dang_path = Path(args.save_path) / f'{mode}' / 'dangerous'
    safe_path = Path(args.save_path) / f'{mode}' / 'safe'
    if args.resize is not None:
        img_transform = transforms.Compose([
            transforms.Resize(size=tuple(args.resize), antialias=True),
        ])

    if mode == 'val' or mode == 'test':
        dang_count = targets.count(1)
        safe_count = targets.count(0)
        max_count = min(dang_count, safe_count)
        reduced_targets = []
        reduced_data_paths = []
        for c in range(2):
            count = 0
            for p, t in zip(data_paths, targets):
                if t == c and count < max_count:
                    count += 1
                    reduced_targets.append(t)
                    reduced_data_paths.append(p)

        data_paths = reduced_data_paths
        targets = reduced_targets

    pbar = tqdm.tqdm(total=len(data_paths), desc=f'Saving {mode} data')

    if targets is not None:
        dang_path.mkdir(parents=True, exist_ok=True)
        safe_path.mkdir(parents=True, exist_ok=True)
        for i, (p, t) in enumerate(zip(data_paths, targets)):
            if t == 1:
                save_path = dang_path / f'{i}.jpg'
            else:
                save_path = safe_path / f'{i}.jpg'
            img = Image.open(p)
            if args.resize is not None:
                img = img_transform(img)
            img.save(save_path)
            pbar.update(1)
    else:
        class_path = Path(args.save_path) / f'{mode}' / 'unlabeled'
        class_path.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(data_paths):
            img = Image.open(p)
            save_path = class_path / f'{i}.jpg'
            if args.resize is not None:
                img = img_transform(img)
            img.save(save_path)
            pbar.update(1)


def make_classes(args, data_paths, mode):
    targets = []
    lb2idx = {
        "person": 1,
        "car": 3,
        "motorcycle": 4,
        "bus": 6,
        "truck": 8
    }
    outputs = detect_targets(args, data_paths, mode)
    for output in outputs:
        labels = output['labels']
        scores = output['scores']
        dangerous = False
        for label, score in zip(labels, scores):
            if label in lb2idx.values() and score > args.conf_threshold:
                dangerous = True
                break
        if dangerous:
            targets.append(1)
        else:
            targets.append(0)
    return targets


if __name__ == '__main__':
    if Path(args.save_path).exists():
        shutil.rmtree(args.save_path)
    random.seed(args.seed)
    train_lb_data, train_ul_data, val_data, test_data = split_data(args)
    train_lb_targets = make_classes(args, train_lb_data, mode='train_lb')
    val_targets = make_classes(args, val_data, mode='val')
    test_targets = make_classes(args, test_data, mode='test')
    save_data(args, train_lb_data, train_lb_targets, mode='train_lb')
    save_data(args, train_ul_data, None, mode='train_ul')
    save_data(args, val_data, val_targets, mode='val')
    save_data(args, test_data, test_targets, mode='test')



