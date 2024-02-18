from typing import Tuple
import tqdm
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image


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
        return img, target
    
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


if __name__ == "__main__":
    train_lb_dataset, train_ul_dataset, val_dataset, test_dataset = get_dreyeve()
    print(train_lb_dataset)
    print(train_ul_dataset)
    print(val_dataset)
    print(test_dataset)