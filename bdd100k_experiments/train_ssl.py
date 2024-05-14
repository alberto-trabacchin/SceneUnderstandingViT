import data
import argparse
from torch.utils.data import DataLoader
from vit_pytorch import SimpleViT, ViT
import torch
import tqdm
import numpy as np
import random
import wandb
import torch.nn.functional as F
from pathlib import Path
from termcolor import colored


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
parser.add_argument('--resize', type=int, default=400, help='(height, width)')
parser.add_argument('--simple', action='store_true')
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


def train_loop(
        args, teacher, student,
        teacher_optim, student_optim,
        teacher_sched, student_sched,
        train_lb_loader, train_ul_loader, val_loader
):
    train_lb_iter = iter(train_lb_loader)
    train_ul_iter = iter(train_ul_loader)
    teacher_val_loss = AverageMeter()
    student_val_loss = AverageMeter()
    teacher_val_acc = AverageMeter()
    student_val_acc = AverageMeter()
    teacher_train_loss = AverageMeter()
    teacher_train_acc = AverageMeter()
    student_train_loss = AverageMeter()
    student_train_acc = AverageMeter()
    teacher_train_f1 = AverageMeter()
    student_train_f1 = AverageMeter()
    teacher_val_f1 = AverageMeter()
    student_val_f1 = AverageMeter()
    teacher_train_prec = AverageMeter()
    student_train_prec = AverageMeter()
    teacher_val_prec = AverageMeter()
    student_val_prec = AverageMeter()
    teacher_train_rec = AverageMeter()
    student_train_rec = AverageMeter()
    teacher_val_rec = AverageMeter()
    student_val_rec = AverageMeter()
    teacher_top1_acc = 0
    student_top1_acc = 0

    pbar = tqdm.tqdm(total=args.eval_steps, position=0, leave=True)

    train_lb_size = train_lb_loader.dataset.__len__()
    train_ul_size = train_ul_loader.dataset.__len__()
    val_size = val_loader.dataset.__len__()
    wandb.init(
        project='SceneUnderstanding',
        name=f'{args.name}_{train_lb_size}LB_{train_ul_size}UL_{val_size}VL',
        config=args
    )

    for step in range(args.train_steps):
        teacher.train()
        student.train()
        try:
            batch_lb = next(train_lb_iter)
        except:
            train_lb_iter = iter(train_lb_loader)
            batch_lb = next(train_lb_iter)
        
        try:
            batch_ul = next(train_ul_iter)
        except:
            train_ul_iter = iter(train_ul_loader)
            batch_ul = next(train_ul_iter)
        
        imgs_lb, targets = batch_lb
        imgs_ul, _ = batch_ul
        imgs_lb, targets = imgs_lb.to(args.device), targets.to(args.device)
        imgs_ul = imgs_ul.to(args.device)
        teacher_optim.zero_grad()
        student_optim.zero_grad()

        teacher_lb_logits = teacher(imgs_lb)
        teacher_ul_logits = teacher(imgs_ul)
        teacher_ul_targets = torch.softmax(teacher_ul_logits, dim=1)

        student_lb_logits = student(imgs_lb)
        student_ul_logits = student(imgs_ul)

        # train the student
        student_optim.zero_grad()
        student_loss = F.cross_entropy(student_lb_logits, targets, reduction="mean")
        student_loss.backward()
        student_grad_1 = [p.grad.data.clone().detach() for p in student.parameters()]

        # train the student
        student_optim.zero_grad()
        student_loss = F.cross_entropy(student_ul_logits, teacher_ul_targets.detach(), reduction="mean")
        student_loss.backward()
        student_grad_2 = [p.grad.data.clone().detach() for p in student.parameters()]
        student_optim.step()
        student_sched.step()

        mpl_coeff = sum([torch.dot(g_1.ravel(), g_2.ravel()).sum().detach().item() for g_1, g_2 in zip(student_grad_1, student_grad_2)])

        # train the teacher
        teacher_optim.zero_grad()
        teacher_loss_ent = F.cross_entropy(teacher_lb_logits, targets, reduction="mean")
        teacher_loss_mpl = mpl_coeff * F.cross_entropy(teacher_ul_logits, teacher_ul_targets.detach(), reduction="mean")

        teacher_loss = teacher_loss_ent + teacher_loss_mpl

        teacher_train_loss.update(teacher_loss.item())
        student_train_loss.update(student_loss.item())
        teacher_train_acc.update(accuracy(teacher_lb_logits, targets))
        student_train_acc.update(accuracy(student_lb_logits, targets))
        teacher_train_prec.update(precision(teacher_lb_logits, targets))
        student_train_prec.update(precision(student_lb_logits, targets))
        teacher_train_rec.update(recall(teacher_lb_logits, targets))
        student_train_rec.update(recall(student_lb_logits, targets))
        teacher_train_f1.update(f1_score(teacher_lb_logits, targets))
        student_train_f1.update(f1_score(student_lb_logits, targets))

        teacher_loss.backward()
        teacher_optim.step()
        teacher_sched.step()
        pbar.update(1)
        pbar.set_description(f"{step+1:4d}/{args.train_steps}  t/t/l: {teacher_train_loss.avg :.4E} | "
                             f"s/t/l: {student_train_loss.avg:.4E} | "
                             f"t/t/a: {teacher_train_acc.avg:.4f} | "
                             f"s/t/a: {student_train_acc.avg:.4f} | "
                             f"t/t/p: {teacher_train_prec.avg:.4f} | "
                             f"s/t/p: {student_train_prec.avg:.4f} | "
                             f"t/t/r: {teacher_train_rec.avg:.4f} | "
                             f"s/t/r: {student_train_rec.avg:.4f} | "
                             f"t/t/f1: {teacher_train_f1.avg:.4f} | "
                             f"s/t/f1: {student_train_f1.avg:.4f}")

        if (step + 1) % args.eval_steps == 0:
            pbar.close()
            pbar = tqdm.tqdm(total=len(val_loader), position=0, leave=True, desc="Validating...")
            teacher.eval()
            student.eval()

            with torch.inference_mode():
                for val_batch in val_loader:
                    imgs, labels = val_batch
                    imgs, labels = imgs.to(args.device), labels.to(args.device)
                    teacher_preds = teacher(imgs)
                    student_preds = student(imgs)
                    teacher_loss = F.cross_entropy(teacher_preds, labels, reduction="mean")
                    student_loss = F.cross_entropy(student_preds, labels, reduction="mean")
                    teacher_val_loss.update(teacher_loss.item())
                    student_val_loss.update(student_loss.item())
                    teacher_val_acc.update(accuracy(teacher_preds, labels))
                    student_val_acc.update(accuracy(student_preds, labels))
                    teacher_val_prec.update(precision(teacher_preds, labels))
                    student_val_prec.update(precision(student_preds, labels))
                    teacher_val_rec.update(recall(teacher_preds, labels))
                    student_val_rec.update(recall(student_preds, labels))
                    teacher_val_f1.update(f1_score(teacher_preds, labels))
                    student_val_f1.update(f1_score(student_preds, labels))
                    pbar.update(1)

            pbar.close()
            
            if teacher_val_acc.avg > teacher_top1_acc:
                teacher_top1_acc = teacher_val_acc.avg
            
            if student_val_acc.avg > student_top1_acc:
                student_top1_acc = student_val_acc.avg
                save_path = Path('checkpoints/')
                save_path.mkdir(parents=True, exist_ok=True)
                save_path = save_path / f'{args.name}_stud.pth'
                torch.save(teacher.state_dict(), save_path)
                wandb.save(f'{args.name}.pth')
                print(colored(f"--> Model saved at {save_path}", "yellow"))

            print(f"teac/valid/loss: {teacher_val_loss.avg:.4E} | teacher/valid/acc: {teacher_val_acc.avg:.4f}")
            print(f"stud/valid/loss: {student_val_loss.avg:.4E} | student/valid/acc: {student_val_acc.avg:.4f}")
            print(f"teac/top1_acc: {teacher_top1_acc:.4f}")
            print(f"stud/top1_acc: {student_top1_acc:.4f}\n")

            wandb.log({
                "teacher/train_loss": teacher_train_loss.avg,
                "student/train_loss": student_train_loss.avg,
                "teacher/val_loss": teacher_val_loss.avg,
                "teacher/val_acc": teacher_val_acc.avg,
                "student/val_loss": student_val_loss.avg,
                "student/val_acc": student_val_acc.avg,
                "teacher/top1_acc": teacher_top1_acc,
                "student/top1_acc": student_top1_acc
            }, step = step)

            teacher_val_loss.reset()
            student_val_loss.reset()
            teacher_val_acc.reset()
            student_val_acc.reset()
            teacher_train_loss.reset()
            teacher_train_acc.reset()
            student_train_loss.reset()
            student_train_acc.reset()

            teacher_train_prec.reset()
            student_train_prec.reset()
            teacher_train_rec.reset()
            student_train_rec.reset()
            teacher_train_f1.reset()
            student_train_f1.reset()
            
            teacher_val_prec.reset()
            student_val_prec.reset()
            teacher_val_rec.reset()
            student_val_rec.reset()
            teacher_val_f1.reset()
            student_val_f1.reset()

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

def precision(preds, labels):
    return (preds.argmax(dim=1) & labels).float().sum() / (preds.argmax(dim=1).float().sum() + 1e-8)

def recall(preds, labels):
    return (preds.argmax(dim=1) & labels).float().sum() / (labels.float().sum() + 1e-8)

def f1_score(preds, labels):
    prec = precision(preds, labels)
    rec = recall(preds, labels)
    return 2 * prec * rec / (prec + rec + 1e-8)



if __name__ == '__main__':
    set_seeds(args.seed)
    train_lb_dataset, val_dataset, train_ul_dataset = data.get_bdd100k(args)
    train_lb_loader = DataLoader(
        dataset = train_lb_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers
    )
    train_ul_loader = DataLoader(
        dataset = train_ul_dataset,
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

    teacher = SimpleViT ( 
        image_size = args.resize,
        patch_size = 15,
        num_classes = args.num_classes,
        dim = 1024,
        depth = 24,
        heads = 16,
        mlp_dim = 4096
    )
    student = SimpleViT ( 
        image_size = args.resize,
        patch_size = 15,
        num_classes = args.num_classes,
        dim = 1024,
        depth = 24,
        heads = 16,
        mlp_dim = 4096
    )
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        teacher = torch.nn.DataParallel(teacher)
        student = torch.nn.DataParallel(student)

    teacher.to(args.device)
    student.to(args.device)
    teacher_optim = torch.optim.Adam(teacher.parameters(), lr=args.lr)
    student_optim = torch.optim.Adam(student.parameters(), lr=args.lr)
    _, _, classes_weights = train_lb_dataset.get_info()

    teacher_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        teacher_optim, 
        T_max=args.train_steps, 
        eta_min=0.0001
    )
    student_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        student_optim, 
        T_max=args.train_steps, 
        eta_min=0.0001
    )


    train_loop(
        args = args,
        teacher = teacher,
        student = student,
        teacher_optim = teacher_optim,
        student_optim = student_optim,
        teacher_sched = teacher_sched,
        student_sched = student_sched,
        train_lb_loader = train_lb_loader,
        train_ul_loader = train_ul_loader,
        val_loader = val_loader
    )