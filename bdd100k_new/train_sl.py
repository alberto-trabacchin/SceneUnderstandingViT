import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
from data import CustomBDD100kDataset
from vit_pytorch import ViT
import wandb
import argparse
import random
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Vision Transformer on BDD100k')
    parser.add_argument("--data-path", type=str, default='/home/alberto/datasets/bdd100k/')
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--name", type=str, default='test_run')
    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                data_loader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                data_loader = val_loader

            running_loss = 0.0
            all_labels = []
            all_preds = []

            # Iterate over data.
            for inputs, labels in tqdm(data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = accuracy_score(all_labels, all_preds)
            epoch_recall = recall_score(all_labels, all_preds, average='binary')
            epoch_precision = precision_score(all_labels, all_preds, average='binary')
            epoch_f1 = f1_score(all_labels, all_preds, average='binary')

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Recall: {epoch_recall:.4f} Precision: {epoch_precision:.4f} F1: {epoch_f1:.4f}')
            
            # Deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), 'best_model.pth')
                print('Best model saved with loss: {:.4f}'.format(best_loss))
            
            wandb.log({f'{phase}_loss': epoch_loss, 
                       f'{phase}_accuracy': epoch_acc, 
                       f'{phase}_recall': epoch_recall, 
                       f'{phase}_precision': epoch_precision, 
                       f'{phase}_f1': epoch_f1}, step=epoch+1)


def vit_l_16_model(weights):
    model = models.vit_l_16(weights = weights)  # Load a pre-trained Vision Transformer
    model.heads = nn.Linear(1024, 2)  # Adjust for binary classification
    return model


if __name__ == '__main__':

    args = parse_args()
    set_seed(42)
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = 1e-3
    num_workers = 4
    model_weights = 'ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1'
    wandb_name = args.name

    vit_params = {
        "image_size": 512,
        "patch_size": 32,
        "num_classes": 2,
        "dim": 512,
        "depth": 12,
        "heads": 8,
        "mlp_dim": 2048,
        "dropout": 0.1,
        "emb_dropout": 0.1
    }


    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="ViT-BDD100k",
        name=wandb_name,

        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "architecture": "ViT",
        "dataset": "BDD100k",
        "epochs": num_epochs,
        "params": vit_params
        }
    )

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((vit_params["image_size"], vit_params["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Assuming you have already defined CustomBDD100kDataset
    train_dataset = CustomBDD100kDataset(
        root_dir=f'{args.data_path}/custom_dataset/train/',  # Update the path accordingly
        transform=transform
    )
    val_dataset = CustomBDD100kDataset(
        root_dir=f'{args.data_path}/custom_dataset/val/',    # Update the path accordingly
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # model = vit_l_16_model(model_weights)
    model = ViT(
        image_size=vit_params["image_size"],
        patch_size=vit_params["patch_size"],
        num_classes=vit_params["num_classes"],
        dim=vit_params["dim"],
        depth=vit_params["depth"],
        heads=vit_params["heads"],
        mlp_dim=vit_params["mlp_dim"],
        dropout=vit_params["dropout"],
        emb_dropout=vit_params["emb_dropout"]
    )

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)  # Example scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-4)  # Example scheduler

    train_model(model, criterion, optimizer, scheduler, num_epochs)