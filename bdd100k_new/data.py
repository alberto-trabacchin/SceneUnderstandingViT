import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomBDD100kDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the subfolders containing images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Load all the images and their labels from subdirectories
        for label_dir in ['class_0', 'class_1']:
            class_label = int(label_dir.split('_')[-1])
            class_dir = os.path.join(root_dir, label_dir)
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(class_dir, img_file), class_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label)



# Define DataLoader Functions
def create_data_loaders(train_dataset, val_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader


if __name__ == "__main__":
    # Define Transformations
    transform = transforms.Compose([
        # transforms.Resize((256, 256)),  # Resize to the same size as the preprocessed images
        transforms.ToTensor(),  # Convert images to tensors
    ])

    # Instantiate Dataset Objects for Train and Validation Sets
    train_dataset = CustomBDD100kDataset(
        root_dir='/home/alberto-trabacchin-wj/datasets/bdd100k/custom_dataset/train/',
        transform=transform
    )

    val_dataset = CustomBDD100kDataset(
        root_dir='/home/alberto-trabacchin-wj/datasets/bdd100k/custom_dataset/val/',
        transform=transform
    )
    # Create Data Loaders
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset)