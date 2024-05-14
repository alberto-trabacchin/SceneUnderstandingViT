from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
from pathlib import Path
from torchvision.transforms import v2
import torch
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./bdd_1k')
    return parser.parse_args()


class BDD100k(ImageFolder):
    def __init__(self, mode, root, lab_size=None, transform=None, target_transform=None):
        self.mode = mode
        self.root = str(root) + '/' + mode
        super(BDD100k, self).__init__(self.root, transform, target_transform)

    def get_info(self):
        classes = self.classes
        classes_count = [0] * len(classes)
        for _, label in self.samples:
            classes_count[label] += 1
        sum_count = sum(classes_count)
        classes_weights = [count / sum_count for count in classes_count]
        assert(sum(classes_weights) == 1)
        return classes, classes_count, classes_weights
    

class BDD100k_UL(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data = list(Path(self.root).rglob('*.jpg'))

    def __getitem__(self, index):
        img = Image.open(self.data[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, 0
    
    def __len__(self):
        return len(self.data)
    

def get_bdd100k(args):
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale = True)
    ])
    train_dataset = BDD100k(root=args.data_path, mode='train', transform=transform)
    unlabeled_dataset = BDD100k_UL(root=args.data_path, transform=transform)
    val_dataset = BDD100k(root=args.data_path, mode='val', transform=transform)
    return train_dataset, val_dataset, unlabeled_dataset
    

if __name__ == "__main__":
    args = parse_args()
    train_dataset, val_dataset, unlabeled_dataset = get_bdd100k(args)
    print(unlabeled_dataset[0])
    
    