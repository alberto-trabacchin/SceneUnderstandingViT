import data
import argparse
from torch.utils.data import DataLoader
from vit_pytorch import SimpleViT
import torch
import tqdm
import numpy as np
import random
import wandb
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='default')
parser.add_argument('--data-path', type=str, default='./dreve')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--train-steps', type=int, default=10000)
parser.add_argument('--eval-steps', type=int, default=500)
parser.add_argument('--num-classes', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--image-size', type=int, nargs='+', default=[216, 384], help='Image size (height, width)')
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


def train_loop(args, model, optimizer, criterion, train_loader, val_loader):
    pbar = tqdm.tqdm(total=args.eval_steps, position=0, leave=True)
    train_lb_size = len(train_lb_dataset)
    test_lb_size = len(test_dataset)
    wandb.init(
        project='DriViSafe-Supervised',
        name=f'{args.name}_{train_lb_size}LB_{test_lb_size}UL',
        config=args
    )
    train_iter = iter(train_loader)
    train_loss = AverageMeter()
    val_loss = AverageMeter()
    train_acc = AverageMeter()
    val_acc = AverageMeter()
    top1_acc = 0

    for step in range(args.train_steps):
        model.train()
        try:
            batch = next(train_iter)
        except:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        imgs, labels, idxs = batch
        imgs, labels = imgs.to(args.device), labels.to(args.device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)
        train_loss.update(loss.item())
        train_acc.update(accuracy(preds, labels))
        loss.backward()
        optimizer.step()
        pbar.update(1)
        pbar.set_description(f"{step+1:4d}/{args.train_steps}  train/loss: {train_loss.avg :.4E} | train/acc: {train_acc.avg:.4f}")
        
        if (step + 1) % args.eval_steps == 0:
            pbar.close()
            pbar = tqdm.tqdm(total=len(val_loader), position=0, leave=True, desc="Validating...")
            model.eval()
            for val_batch in val_loader:
                imgs, labels, idxs = val_batch
                imgs, labels = imgs.to(args.device), labels.to(args.device)
                with torch.inference_mode():
                    preds = model(imgs)
                    loss = criterion(preds, labels)
                    val_loss.update(loss.item())
                    val_acc.update(accuracy(preds, labels))
                pbar.update(1)
            pbar.set_description(f"{step+1:4d}/{args.train_steps}  VALID/loss: {val_loss.avg:.4E} | VALID/acc: {val_acc.avg:.4f}")
            if val_acc.avg > top1_acc:
                top1_acc = val_acc.avg
                save_path = Path('checkpoints/')
                save_path.mkdir(parents=True, exist_ok=True)
                save_path = save_path / f'{args.name}.pth'
                torch.save(model.state_dict(), save_path)
                wandb.save(f'{args.name}.pth')
                print(f"--> Model saved at {save_path}")
            wandb.log({
                "train/loss": train_loss.avg,
                "train/acc": train_acc.avg,
                "val/loss": val_loss.avg,
                "val/acc": val_acc.avg,
                "top1_acc": top1_acc
            }, step = step)
            wandb.watch(models = model, log='all')
            print(f'top1_acc: {top1_acc:.6f}')
            val_loss.reset()
            val_acc.reset()
            train_loss.reset()
            train_acc.reset()
            pbar.close()
            pbar = tqdm.tqdm(total=args.eval_steps, position=0, leave=True)


def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def accuracy(preds, labels):
    return (preds.argmax(dim=1) == labels).float().mean()



if __name__ == '__main__':
    train_lb_dataset, train_ul_dataset, val_dataset, test_dataset = data.get_dreyeve(args)
    train_lb_loader = DataLoader(
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
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.workers
    )

    model = SimpleViT(
        image_size = tuple(args.image_size),
        patch_size = 6,
        num_classes = args.num_classes,
        dim = 64,
        depth = 6,
        heads = 8,
        mlp_dim = 128
    )
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    _, _, classes_weights = train_lb_dataset.get_info()
    criterion = torch.nn.CrossEntropyLoss(
        weight = torch.tensor(classes_weights).to(args.device)
    )

    train_loop(
        args = args,
        model = model,
        optimizer = optimizer,
        criterion = criterion,
        train_loader = train_lb_loader,
        val_loader = val_loader
    )