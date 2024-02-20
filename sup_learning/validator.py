import torchvision
import torch
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torch.utils.data import DataLoader
import argparse
import wandb
import data
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='./data')
parser.add_argument('--model-path', type=str, default='./wandb/latest-run')
parser.add_argument('--data-count', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()


def validate(args, train_ul_dataset):
    validator = retinanet_resnet50_fpn(weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT)
    validator.to(args.device)
    validator.eval()
    model = wandb.restore('model.pt', run_path=args.model_path)
    model.load_weights(model.name)
    model.to(args.device)
    model.eval()
    loader = DataLoader(
        dataset=train_ul_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )
    preds_per_class = args.data_count // 2
    data_counter = [0, 0]
    val_idxs = []
    val_targets = []

    for batch in loader:
        images, _, idxs = batch            
        with torch.inference_mode():
            images = images.to(args.device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            targets = validator(images)
            equal_elements = preds.eq(targets)
            indexes = torch.nonzero(equal_elements)
        
        for c in range(0, 2):
            if data_counter[c] >= preds_per_class:
                continue
            if data_counter[0] >= preds_per_class and data_counter[1] >= preds_per_class:
                break
            new_idxs = [idxs[i] for i in indexes if targets[i] == c]
            new_targets = [targets[i] for i in indexes if targets[i] == c]
            val_idxs.extend(new_idxs)
            val_targets.extend(new_targets)
            data_counter[c] += len(new_idxs)

    return val_idxs, val_targets


def update_train_dataset(args, val_idxs, val_targets, train_ul_dataset):
    val_dataset = train_ul_dataset[val_idxs]
    val_dataset.targets = val_targets
    
    for p in val_dataset.samples:
        img_path = p[0]
        target = p[1]
        print(img_path)



            







if __name__ == "__main__":
    _, train_ul_dataset, _, _ = data.get_dreyeve(args)
    val_idxs, val_targets = validate(args, train_ul_dataset)
    update_train_dataset(args, val_idxs, val_targets, train_ul_dataset)
