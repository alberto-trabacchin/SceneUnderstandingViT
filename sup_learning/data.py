from typing import Tuple
import tqdm
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
from nuimages import NuImages
import os
import matplotlib.pyplot as plt
import json
import argparse
import random


parser = argparse.ArgumentParser()
parser.add_argument("--data-path", type=str, default="./data")
args = parser.parse_args()



class NuDataset(Dataset):
    def _make_labels(self, nuim, dang_classes):
        labels = {
            "safe": [],
            "dangerous": []
        }
        paths = {
            "safe": [],
            "dangerous": []
        }
        pbar = tqdm.tqdm(
            total = len(nuim.sample), 
            desc = f"Preparing annotations for {self.version}..."
        )

        for s in nuim.sample:
            sd_token = s["key_camera_token"]
            obj_annots = [o for o in nuim.object_ann if o["sample_data_token"] == sd_token]
            cat_names = [nuim.get("category", o["category_token"])["name"] for o in obj_annots]
            is_dangerous = False
            for cn in cat_names:
                if any(c in cn for c in dang_classes):
                    labels["dangerous"].append(sd_token)
                    is_dangerous = True
                    break
            if not is_dangerous:
                labels["safe"].append(sd_token)
            pbar.update(1)
        pbar.close()
        print("Annotations saved to: ", self.labels_path)

        for sd_token in labels["safe"]:
            sample_data = nuim.get("sample_data", sd_token)
            image_path = os.path.join(nuim.dataroot, sample_data["filename"])
            paths["safe"].append(image_path)
        
        for sd_token in labels["dangerous"]:
            sample_data = nuim.get("sample_data", sd_token)
            image_path = os.path.join(nuim.dataroot, sample_data["filename"])
            paths["dangerous"].append(image_path)
        
        with open(self.labels_path, "w") as f:
            json.dump(paths, f)        
            

    def __init__(self, version, root, transform = None, target_transform = None):
        self.dang_classes = [
            "human",
            "vehicle",
            "static_object.bicycle_rack"
        ]
        self.idx2label = {
            0: "safe",
            1: "dangerous"
        }
        
        self.version = version
        self.transform = transform
        self.target_transform = target_transform
        self.nuim = NuImages(
            dataroot=root, 
            version=version, 
            verbose=False, 
            lazy=True
        )

        self.labels_path = str(root) + f"/{version}/targets.json"

        if not os.path.exists(self.labels_path):
            self._make_labels(self.nuim, self.dang_classes)

        with open(self.labels_path, "r") as f:
            self.data_path = json.load(f)
        
        self.data = self.data_path["safe"] + self.data_path["dangerous"]
        self.targets = [0] * len(self.data) + [1] * len(self.data_path["dangerous"])
        print(len(self.data_path["safe"]))
        print(len(self.data_path["dangerous"]))
        print(len(self.nuim.sample))
        exit()

        tmp = list(zip(self.data, self.targets))
        random.shuffle(tmp)
        self.data, self.targets = zip(*tmp)


    def __getitem__(self, index: int):
        sample = self.data[index]
        image = Image.open(sample)
        target = self.targets[index]

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target
    

    def __len__(self):
        return len(self.nuim.sample)


class Dreyeve(ImageFolder):
    def __init__(self, mode, root, transform=None, target_transform=None, loader=None):
        self.mode = mode
        self.root = str(root) + '/' + mode
        super(Dreyeve, self).__init__(self.root, transform, target_transform, loader)

    def __getitem__(self, index: int):
        img_path, target = self.samples[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index
    
    def get_info(self):
        classes = self.classes
        classes_count = [0] * len(classes)
        for _, label in self.samples:
            classes_count[label] += 1
        sum_count = sum(classes_count)
        classes_weights = [count / sum_count for count in classes_count]
        assert(sum(classes_weights) == 1)
        return classes, classes_count, classes_weights

        
def get_dreyeve(args):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_lb_dataset = Dreyeve(root=args.data_path, mode='train_lb', transform=transform)
    train_ul_dataset = Dreyeve(root=args.data_path, mode='train_ul', transform=transform)
    val_dataset = Dreyeve(root=args.data_path, mode='val', transform=transform)
    test_dataset = Dreyeve(root=args.data_path, mode='test', transform=transform)
    return train_lb_dataset, train_ul_dataset, val_dataset, test_dataset


def get_nuimages(args):
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    train_dataset = NuDataset(version="v1.0-train", root=args.data_path, transform=transform)
    val_dataset = NuDataset(version="v1.0-val", root=args.data_path, transform=transform)
    return train_dataset, val_dataset


if __name__ == "__main__":
    train_dataset, val_dataset = get_nuimages(args)