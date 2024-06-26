import argparse
import torch
import tqdm
import wandb
from pathlib import Path
from termcolor import colored
import matplotlib.pyplot as plt
from nuimages import NuImages
import numpy as np
import random
from torch.utils.data import DataLoader
import data
from vit_pytorch import SimpleViT


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='default')
parser.add_argument('--data-path', type=str, default='/home/alberto/datasets/nuimages')
parser.add_argument('--clear-labels', action='store_true')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--train-steps', type=int, default=10000)
parser.add_argument('--eval-steps', type=int, default=500)
parser.add_argument('--num-classes', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--resize', type=int, required=True, help='Image size (height, width)')
args = parser.parse_args()


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_loop(args, model, optimizer, criterion, train_loader, val_loader, scheduler):
    # pbar = tqdm.tqdm(total=args.eval_steps, position=0, leave=True)
    train_lb_size = train_loader.dataset.__len__()
    val_size = val_loader.dataset.__len__()
    wandb.init(
        project='SceneUnderstanding',
        name=f'{args.name}_{train_lb_size}LB_{val_size}VL',
        config=args
    )
    train_iter = iter(train_loader)
    train_loss = AverageMeter()
    val_loss = AverageMeter()
    train_acc = AverageMeter()
    val_acc = AverageMeter()
    train_prec = AverageMeter()
    val_prec = AverageMeter()
    train_rec = AverageMeter()
    val_rec = AverageMeter()
    train_f1 = AverageMeter()
    val_f1 = AverageMeter()
    top1_acc = 0
    top_f1 = 0
    model.train()

    # for step in range(args.train_steps):
    epochs = int(100e3)
    for epoch in range(epochs):
        pbar = tqdm.tqdm(total=len(train_loader), position=0, leave=True, desc="Training...")
        for batch in train_loader:
            # model.train()
            # try:
            #     batch = next(train_iter)
            # except:
            #     train_iter = iter(train_loader)
            #     batch = next(train_iter)

            imgs, labels = batch
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            train_loss.update(loss.item())
            acc_res, train_cm = accuracy(preds, labels)
            train_acc.update(acc_res)
            train_prec.update(precision(preds, labels))
            train_rec.update(recall(preds, labels))
            train_f1.update(f1_score(preds, labels))
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.update(1)
            pbar.set_description(f"{epoch+1:4d}/{epochs}  train/loss: {train_loss.avg :.4E} | "
                                f"train/acc: {train_acc.avg:.4f} | "
                                f"train/prec: {train_prec.avg:.4f} | "
                                f"train/rec: {train_rec.avg:.4f} | "
                                f"train/f1: {train_f1.avg:.4f}")
        pbar.close()
        
        # if (step + 1) % args.eval_steps == 0:
            # pbar.close()
        pbar = tqdm.tqdm(total=len(val_loader), position=0, leave=True, desc="Validating...")
        model.eval()
        for val_batch in val_loader:
            imgs, labels = val_batch
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            with torch.inference_mode():
                preds = model(imgs)
                loss = criterion(preds, labels)
                val_loss.update(loss.item())
                acc_res, val_cm = accuracy(preds, labels)
                val_acc.update(acc_res)
                val_prec.update(precision(preds, labels))
                val_rec.update(recall(preds, labels))
                val_f1.update(f1_score(preds, labels))
            pbar.update(1)
        pbar.set_description(f"{epoch+1:4d}/{epochs}  VALID/loss: {val_loss.avg:.4E} | "
                                f"VALID/acc: {val_acc.avg:.4f} | "
                                f"VALID/prec: {val_prec.avg:.4f} | "
                                f"VALID/rec: {val_rec.avg:.4f} | "
                                f"VALID/f1: {val_f1.avg:.4f}")
        pbar.close()
        if val_f1.avg > top_f1:
            top_f1 = val_f1.avg
            save_path = Path('checkpoints/')
            save_path.mkdir(parents=True, exist_ok=True)
            save_path = save_path / f'{args.name}.pth'
            torch.save(model.state_dict(), save_path)
            wandb.save(f'{args.name}.pth')
            print(colored(f"--> Model saved at {save_path}", "yellow"))
        # wandb.log({
        #     "train/loss": train_loss.avg,
        #     "train/acc": train_acc.avg,
        #     "val/loss": val_loss.avg,
        #     "val/acc": val_acc.avg,
        #     "top1_acc": top1_acc
        # }, step = step)
        # wandb.watch(models = model, log='all')
        print(f'top_f1: {top_f1:.6f}\n')
        val_loss.reset()
        val_acc.reset()
        train_loss.reset()
        train_acc.reset()
        train_prec.reset()
        val_prec.reset()
        train_rec.reset()
        val_rec.reset()
        train_f1.reset()
        val_f1.reset()
        # pbar = tqdm.tqdm(total=args.eval_steps, position=0, leave=True)


def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def accuracy(preds, labels):
    tp = (preds.argmax(dim=1) & labels).float().sum()
    tn = ((1 - preds.argmax(dim=1)) & (1 - labels)).float().sum()
    fp = (preds.argmax(dim=1) & (1 - labels)).float().sum()
    fn = ((1 - preds.argmax(dim=1)) & labels).float().sum()
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    conf_mat = {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }
    return acc, conf_mat

def precision(preds, labels):
    tp = (preds.argmax(dim=1) & labels).float().sum()
    fp = (preds.argmax(dim=1) & (1 - labels)).float().sum()
    return tp / (tp + fp + 1e-8)

def recall(preds, labels):
    tp = (preds.argmax(dim=1) & labels).float().sum()
    fn = ((1 - preds.argmax(dim=1)) & labels).float().sum()
    return tp / (tp + fn + 1e-8)

def f1_score(preds, labels):
    prec = precision(preds, labels)
    rec = recall(preds, labels)
    return 2 * prec * rec / (prec + rec + 1e-8)


if __name__ == '__main__':
    set_seeds(args.seed)
    # train_lb_dataset, train_ul_dataset, val_dataset, test_dataset = data.get_dreyeve(args)
    
    train_lb_dataset, val_dataset, test_dataset, _ = data.get_bdd100k(args)
    train_loader = DataLoader(
        dataset = train_lb_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers
    )
    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.workers
    )

    model = SimpleViT ( 
        image_size = args.resize,
        patch_size = 20,
        num_classes = 2,
        dim = 512,
        depth = 4,
        heads = 4,
        mlp_dim = 512
    )

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    _, _, classes_weights = train_lb_dataset.get_info()
    criterion = torch.nn.CrossEntropyLoss(
        weight = torch.tensor(classes_weights).to(args.device)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.train_steps, 
        eta_min=0.0001
    )

    train_loop(
        args = args,
        model = model,
        optimizer = optimizer,
        criterion = criterion,
        train_loader = train_loader,
        val_loader = val_loader,
        scheduler = scheduler
    )