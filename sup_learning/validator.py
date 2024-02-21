import torchvision
import torch
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torch.utils.data import DataLoader
import argparse
import wandb
import data
from pathlib import Path
from vit_pytorch import SimpleViT
import numpy as np
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='./data')
parser.add_argument('--model-path', type=str, default='./wandb/latest-run')
parser.add_argument('--data-count', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--image-size', type=int, nargs='+', default=[216, 384], help='(height, width)')
parser.add_argument('--num-classes', type=int, default=2)
parser.add_argument('--conf-threshold', type=float, default=0.7)
parser.add_argument('--validation-name', type=str, default='default')
args = parser.parse_args()


def outputs2targets(outputs, conf_threshold):
    targets = []
    lb2idx = {
        "person": 1,
        "car": 3,
        "motorcycle": 4,
        "bus": 6,
        "truck": 8
    }
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
    return torch.tensor(targets)


def validate(args, train_ul_dataset, train_lb_dataset):

    def counter_limit(counter, limit):
        lim_safe = counter[0] >= limit
        lim_danger = counter[1] >= limit
        return lim_safe and lim_danger
    
    lb_names = [Path(s).name for s, _ in train_lb_dataset.samples]
    ul_names = [Path(s).name for s, _ in train_ul_dataset.samples]
    new_ul_idxs = []

    for i, s in enumerate(ul_names):
        if s not in lb_names:
            new_ul_idxs.append(i)
    train_ul_dataset.samples = [train_ul_dataset.samples[i] for i in new_ul_idxs]
    train_ul_dataset.targets = [train_ul_dataset.targets[i] for i in new_ul_idxs]
    
    validator = retinanet_resnet50_fpn(weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT)
    validator.to(args.device)
    validator.eval()
    model = SimpleViT(
        image_size = tuple(args.image_size),
        patch_size = 6,
        num_classes = args.num_classes,
        dim = 64,
        depth = 6,
        heads = 8,
        mlp_dim = 128
    )
    model.load_state_dict(torch.load(args.model_path))
    model.to(args.device)
    
    loader = DataLoader(
        dataset=train_ul_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )
    preds_per_class = args.data_count // 2
    print(f'Validating {preds_per_class} samples per class')
    data_counter = [0, 0]
    val_idxs = []
    val_targets = []

    for batch in loader:
        if counter_limit(data_counter, preds_per_class):
            break

        images, _, idxs = batch
        with torch.inference_mode():
            images = images.to(args.device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).to('cpu')
            outputs = validator(images)
            targets = outputs2targets(outputs, args.conf_threshold)
            equal_elements = preds.eq(targets)
            wrng_idxs = torch.nonzero(equal_elements)
        
        for c in range(0, 2):
            if data_counter[c] >= preds_per_class:
                continue
            new_idxs = [idxs[i] for i in wrng_idxs if targets[i] == c]
            new_targets = [targets[i] for i in wrng_idxs if targets[i] == c]
            val_idxs.extend(new_idxs)
            val_targets.extend(new_targets)
            data_counter[c] += len(new_idxs)
        print(data_counter)

    val_idxs = [t.item() for t in val_idxs]
    val_targets = [t.item() for t in val_targets]

    return val_idxs, val_targets


def update_train_dataset(args, val_idxs, val_targets, train_ul_dataset, train_lb_dataset):
    val_idxs = np.array(val_idxs, dtype = np.int32)
    val_paths = [train_ul_dataset.samples[i][0] for i in val_idxs]
    data_path = args.data_path
    pred_path = data_path + '/predictions/' + args.validation_name
    if Path(pred_path).exists():
        shutil.rmtree(pred_path)
    Path(pred_path).mkdir(exist_ok=True, parents=True)
    Path(pred_path + '/dangerous').mkdir(exist_ok=True, parents=True)
    Path(pred_path + '/safe').mkdir(exist_ok=True, parents=True)

    for dp, t in zip(val_paths, val_targets):
        fname = Path(dp).name
        if t == 0:
            shutil.copy(dp, pred_path + f'/dangerous/{str(fname)}')
            shutil.move(dp, train_lb_dataset.root + '/safe')
            print(f'{dp} moved to {train_lb_dataset.root}/safe')
        elif t == 1:
            shutil.copy(dp, pred_path + '/safe')
            shutil.move(dp, train_lb_dataset.root + '/dangerous')
            print(f'{dp} moved to {train_lb_dataset.root}/dangerous')
        else:
            raise ValueError(f'Invalid target: {t}')


if __name__ == "__main__":
    train_lb_dataset, train_ul_dataset, _, _ = data.get_dreyeve(args)
    val_idxs, val_targets = validate(args, train_ul_dataset, train_lb_dataset)
    update_train_dataset(args, val_idxs, val_targets, train_ul_dataset, train_lb_dataset)
