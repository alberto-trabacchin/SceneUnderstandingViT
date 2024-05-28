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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from termcolor import colored
from torchvision.transforms import transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Vision Transformer on BDD100k')
    parser.add_argument("--data-path", type=str, default='/home/alberto/datasets/bdd100k/')
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--name", type=str, default='test_run')
    args = parser.parse_args()
    return args


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
    # train_lb_iter = iter(train_lb_loader)
    train_ul_iter = iter(train_ul_loader)
    teach_run_loss = 0.0
    stud_run_loss = 0.0

    for step in range(args.epochs):
        teacher.train()
        student.train()
        all_labels = []
        teach_all_preds = []
        stud_all_preds = []
        print(f"Epoch {step+1}/{args.epochs}")
        print("-" * 10)
        pbar = tqdm.tqdm(total=len(train_lb_loader), position=0, leave=True)
        for batch_lb in train_lb_loader:
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

            teacher_loss.backward()
            teacher_optim.step()
            teacher_sched.step()

            all_labels.extend(targets.cpu().numpy())
            teach_all_preds.extend(teacher_lb_logits.argmax(dim=1).cpu().numpy())
            stud_all_preds.extend(student_lb_logits.argmax(dim=1).cpu().numpy())
            teach_run_loss += teacher_loss.item() * imgs_lb.size(0)
            stud_run_loss += student_loss.item() * imgs_lb.size(0)
            pbar.update(1)

        pbar.close()

        teach_train_loss = teach_run_loss / len(train_lb_loader.dataset)
        stud_train_loss = stud_run_loss / len(train_lb_loader.dataset)
        teach_train_acc = accuracy_score(all_labels, teach_all_preds)
        stud_train_acc = accuracy_score(all_labels, stud_all_preds)
        teach_train_prec = precision_score(all_labels, teach_all_preds)
        stud_train_prec = precision_score(all_labels, stud_all_preds)
        teach_train_rec = recall_score(all_labels, teach_all_preds)
        stud_train_rec = recall_score(all_labels, stud_all_preds)
        teach_train_f1 = f1_score(all_labels, teach_all_preds)
        stud_train_f1 = f1_score(all_labels, stud_all_preds)

        print(f"T/train/loss: {teach_train_loss:.4E} | T/train/acc: {teach_train_acc:.4f} | T/train/prec: {teach_train_prec:.4f} | T/train/rec: {teach_train_rec:.4f} | T/train/f1: {teach_train_f1:.4f}")
        print(f"S/train/loss: {stud_train_loss:.4E} | S/train/acc: {stud_train_acc:.4f} | S/train/prec: {stud_train_prec:.4f} | S/train/rec: {stud_train_rec:.4f} | S/train/f1: {stud_train_f1:.4f}")
        
        # Evaluation
        teacher.eval()
        student.eval()

        pbar = tqdm.tqdm(total=len(val_loader), position=0, leave=True)
        teach_run_loss = 0.0
        stud_run_loss = 0.0
        all_labels = []
        teach_all_preds = []
        stud_all_preds = []
        with torch.inference_mode():
            for batch in val_loader:
                imgs, labels = batch
                imgs, labels = imgs.to(args.device), labels.to(args.device)
                teacher_preds = teacher(imgs)
                student_preds = student(imgs)
                teacher_loss = F.cross_entropy(teacher_preds, labels, reduction="mean")
                student_loss = F.cross_entropy(student_preds, labels, reduction="mean")

                all_labels.extend(labels.cpu().numpy())
                teach_all_preds.extend(teacher_preds.argmax(dim=1).cpu().numpy())
                stud_all_preds.extend(student_preds.argmax(dim=1).cpu().numpy())
                teach_run_loss += teacher_loss.item() * imgs.size(0)
                stud_run_loss += student_loss.item() * imgs.size(0)
                
                pbar.update(1)
        pbar.close()

        teacher_val_loss = teach_run_loss / len(val_loader.dataset)
        student_val_loss = stud_run_loss / len(val_loader.dataset)
        teacher_val_acc = accuracy_score(all_labels, teach_all_preds)
        student_val_acc = accuracy_score(all_labels, stud_all_preds)
        teacher_val_prec = precision_score(all_labels, teach_all_preds)
        student_val_prec = precision_score(all_labels, stud_all_preds)
        teacher_val_rec = recall_score(all_labels, teach_all_preds)
        student_val_rec = recall_score(all_labels, stud_all_preds)
        teacher_val_f1 = f1_score(all_labels, teach_all_preds)
        student_val_f1 = f1_score(all_labels, stud_all_preds)

        print(f"T/val/loss: {teacher_val_loss:.4E} | T/val/acc: {teacher_val_acc:.4f} | T/val/prec: {teacher_val_prec:.4f} | T/val/rec: {teacher_val_rec:.4f} | T/val/f1: {teacher_val_f1:.4f}")
        print(f"S/val/loss: {student_val_loss:.4E} | S/val/acc: {student_val_acc:.4f} | S/val/prec: {student_val_prec:.4f} | S/val/rec: {student_val_rec:.4f} | S/val/f1: {student_val_f1:.4f}")

        wandb.log({
            "teacher/train_loss": teach_train_loss,
            "student/train_loss": stud_train_loss,
            "teacher/val_loss": teacher_val_loss,
            "teacher/val_acc": teacher_val_acc,
            "student/val_loss": student_val_loss,
            "student/val_acc": student_val_acc,
            "teacher/val_prec": teacher_val_prec,
            "student/val_prec": student_val_prec,
            "teacher/val_rec": teacher_val_rec,
            "student/val_rec": student_val_rec,
            "teacher/val_f1": teacher_val_f1,
            "student/val_f1": student_val_f1
            }, step = step)    



# def train_loop(
#         args, teacher, student,
#         teacher_optim, student_optim,
#         teacher_sched, student_sched,
#         train_lb_loader, train_ul_loader, val_loader
# ):
#     train_lb_iter = iter(train_lb_loader)
#     train_ul_iter = iter(train_ul_loader)
#     teacher_val_loss = AverageMeter()
#     student_val_loss = AverageMeter()
#     teacher_val_acc = AverageMeter()
#     student_val_acc = AverageMeter()
#     teacher_train_loss = AverageMeter()
#     teacher_train_acc = AverageMeter()
#     student_train_loss = AverageMeter()
#     student_train_acc = AverageMeter()
#     teacher_train_f1 = AverageMeter()
#     student_train_f1 = AverageMeter()
#     teacher_val_f1 = AverageMeter()
#     student_val_f1 = AverageMeter()
#     teacher_train_prec = AverageMeter()
#     student_train_prec = AverageMeter()
#     teacher_val_prec = AverageMeter()
#     student_val_prec = AverageMeter()
#     teacher_train_rec = AverageMeter()
#     student_train_rec = AverageMeter()
#     teacher_val_rec = AverageMeter()
#     student_val_rec = AverageMeter()
#     teacher_top1_acc = 0
#     student_top1_acc = 0

#     pbar = tqdm.tqdm(total=args.eval_steps, position=0, leave=True)

#     train_lb_size = train_lb_loader.dataset.__len__()
#     train_ul_size = train_ul_loader.dataset.__len__()
#     val_size = val_loader.dataset.__len__()
#     wandb.init(
#         project='SceneUnderstanding',
#         name=f'{args.name}_{train_lb_size}LB_{train_ul_size}UL_{val_size}VL',
#         config=args
#     )

#     for step in range(args.train_steps):
#         teacher.train()
#         student.train()
#         try:
#             batch_lb = next(train_lb_iter)
#         except:
#             train_lb_iter = iter(train_lb_loader)
#             batch_lb = next(train_lb_iter)
        
#         try:
#             batch_ul = next(train_ul_iter)
#         except:
#             train_ul_iter = iter(train_ul_loader)
#             batch_ul = next(train_ul_iter)
        
#         imgs_lb, targets = batch_lb
#         imgs_ul, _ = batch_ul
#         imgs_lb, targets = imgs_lb.to(args.device), targets.to(args.device)
#         imgs_ul = imgs_ul.to(args.device)
#         teacher_optim.zero_grad()
#         student_optim.zero_grad()

#         teacher_lb_logits = teacher(imgs_lb)
#         teacher_ul_logits = teacher(imgs_ul)
#         teacher_ul_targets = torch.softmax(teacher_ul_logits, dim=1)

#         student_lb_logits = student(imgs_lb)
#         student_ul_logits = student(imgs_ul)

#         # train the student
#         student_optim.zero_grad()
#         student_loss = F.cross_entropy(student_lb_logits, targets, reduction="mean")
#         student_loss.backward()
#         student_grad_1 = [p.grad.data.clone().detach() for p in student.parameters()]

#         # train the student
#         student_optim.zero_grad()
#         student_loss = F.cross_entropy(student_ul_logits, teacher_ul_targets.detach(), reduction="mean")
#         student_loss.backward()
#         student_grad_2 = [p.grad.data.clone().detach() for p in student.parameters()]
#         student_optim.step()
#         student_sched.step()

#         mpl_coeff = sum([torch.dot(g_1.ravel(), g_2.ravel()).sum().detach().item() for g_1, g_2 in zip(student_grad_1, student_grad_2)])

#         # train the teacher
#         teacher_optim.zero_grad()
#         teacher_loss_ent = F.cross_entropy(teacher_lb_logits, targets, reduction="mean")
#         teacher_loss_mpl = mpl_coeff * F.cross_entropy(teacher_ul_logits, teacher_ul_targets.detach(), reduction="mean")

#         teacher_loss = teacher_loss_ent + teacher_loss_mpl

#         teacher_train_loss.update(teacher_loss.item())
#         student_train_loss.update(student_loss.item())
#         teacher_train_acc.update(accuracy(teacher_lb_logits, targets))
#         student_train_acc.update(accuracy(student_lb_logits, targets))
#         teacher_train_prec.update(precision(teacher_lb_logits, targets))
#         student_train_prec.update(precision(student_lb_logits, targets))
#         teacher_train_rec.update(recall(teacher_lb_logits, targets))
#         student_train_rec.update(recall(student_lb_logits, targets))
#         teacher_train_f1.update(f1_score(teacher_lb_logits, targets))
#         student_train_f1.update(f1_score(student_lb_logits, targets))

#         teacher_loss.backward()
#         teacher_optim.step()
#         teacher_sched.step()
#         pbar.update(1)
#         pbar.set_description(f"{step+1:4d}/{args.train_steps}  t/t/l: {teacher_train_loss.avg :.4E} | "
#                              f"s/t/l: {student_train_loss.avg:.4E} | "
#                              f"t/t/a: {teacher_train_acc.avg:.4f} | "
#                              f"s/t/a: {student_train_acc.avg:.4f} | "
#                              f"t/t/p: {teacher_train_prec.avg:.4f} | "
#                              f"s/t/p: {student_train_prec.avg:.4f} | "
#                              f"t/t/r: {teacher_train_rec.avg:.4f} | "
#                              f"s/t/r: {student_train_rec.avg:.4f} | "
#                              f"t/t/f1: {teacher_train_f1.avg:.4f} | "
#                              f"s/t/f1: {student_train_f1.avg:.4f}")

#         if (step + 1) % args.eval_steps == 0:
#             pbar.close()
#             pbar = tqdm.tqdm(total=len(val_loader), position=0, leave=True, desc="Validating...")
#             teacher.eval()
#             student.eval()

#             with torch.inference_mode():
#                 for val_batch in val_loader:
#                     imgs, labels = val_batch
#                     imgs, labels = imgs.to(args.device), labels.to(args.device)
#                     teacher_preds = teacher(imgs)
#                     student_preds = student(imgs)
#                     teacher_loss = F.cross_entropy(teacher_preds, labels, reduction="mean")
#                     student_loss = F.cross_entropy(student_preds, labels, reduction="mean")
#                     teacher_val_loss.update(teacher_loss.item())
#                     student_val_loss.update(student_loss.item())
#                     teacher_val_acc.update(accuracy(teacher_preds, labels))
#                     student_val_acc.update(accuracy(student_preds, labels))
#                     teacher_val_prec.update(precision(teacher_preds, labels))
#                     student_val_prec.update(precision(student_preds, labels))
#                     teacher_val_rec.update(recall(teacher_preds, labels))
#                     student_val_rec.update(recall(student_preds, labels))
#                     teacher_val_f1.update(f1_score(teacher_preds, labels))
#                     student_val_f1.update(f1_score(student_preds, labels))
#                     pbar.update(1)

#             pbar.close()
            
#             if teacher_val_acc.avg > teacher_top1_acc:
#                 teacher_top1_acc = teacher_val_acc.avg
            
#             if student_val_acc.avg > student_top1_acc:
#                 student_top1_acc = student_val_acc.avg
#                 save_path = Path('checkpoints/')
#                 save_path.mkdir(parents=True, exist_ok=True)
#                 save_path = save_path / f'{args.name}_stud.pth'
#                 torch.save(teacher.state_dict(), save_path)
#                 wandb.save(f'{args.name}.pth')
#                 print(colored(f"--> Model saved at {save_path}", "yellow"))

#             print(f"teac/valid/loss: {teacher_val_loss.avg:.4E} | teacher/valid/acc: {teacher_val_acc.avg:.4f}")
#             print(f"stud/valid/loss: {student_val_loss.avg:.4E} | student/valid/acc: {student_val_acc.avg:.4f}")
#             print(f"teac/top1_acc: {teacher_top1_acc:.4f}")
#             print(f"stud/top1_acc: {student_top1_acc:.4f}\n")

#             wandb.log({
#                 "teacher/train_loss": teacher_train_loss.avg,
#                 "student/train_loss": student_train_loss.avg,
#                 "teacher/val_loss": teacher_val_loss.avg,
#                 "teacher/val_acc": teacher_val_acc.avg,
#                 "student/val_loss": student_val_loss.avg,
#                 "student/val_acc": student_val_acc.avg,
#                 "teacher/top1_acc": teacher_top1_acc,
#                 "student/top1_acc": student_top1_acc
#             }, step = step)

#             teacher_val_loss.reset()
#             student_val_loss.reset()
#             teacher_val_acc.reset()
#             student_val_acc.reset()
#             teacher_train_loss.reset()
#             teacher_train_acc.reset()
#             student_train_loss.reset()
#             student_train_acc.reset()

#             teacher_train_prec.reset()
#             student_train_prec.reset()
#             teacher_train_rec.reset()
#             student_train_rec.reset()
#             teacher_train_f1.reset()
#             student_train_f1.reset()
            
#             teacher_val_prec.reset()
#             student_val_prec.reset()
#             teacher_val_rec.reset()
#             student_val_rec.reset()
#             teacher_val_f1.reset()
#             student_val_f1.reset()

#             pbar = tqdm.tqdm(total=args.eval_steps, position=0, leave=True)
    

def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)



if __name__ == '__main__':
    args = parse_args()
    set_seeds(42)
    # Device configuration
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = 1e-3
    num_workers = 4
    model_weights = 'ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1'
    wandb_name = args.name + "_SSL"

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

    # For pretrained model
    # vit_params = {
    #     "image_size": 512,
    #     "patch_size": 32,
    #     "num_classes": 2,
    #     "dim": 1024,
    #     "depth": 24,
    #     "heads": 16,
    #     "mlp_dim": 4096,
    #     "dropout": 0.1,
    #     "emb_dropout": 0.1
    # }


    transform = transforms.Compose([
        transforms.Resize((vit_params["image_size"], vit_params["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_lb_dataset = data.CustomBDD100kDataset(
        root_dir=f'{args.data_path}/custom_dataset/train/',  # Update the path accordingly
        transform=transform
    )
    train_ul_dataset = data.CustomBDD100kDataset(
        root_dir=f'{args.data_path}/custom_dataset/unlabeled/',    # Update the path accordingly
        transform=transform,
        labeled=False
    )
    val_dataset = data.CustomBDD100kDataset(
        root_dir=f'{args.data_path}/custom_dataset/val/',    # Update the path accordingly
        transform=transform
    )
    print(f"Train LB: {len(train_lb_dataset)} | Train UL: {len(train_ul_dataset)} | Val: {len(val_dataset)}")
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
        "params": vit_params,
        "ul_size": len(train_ul_dataset),
        "lb_size": len(train_lb_dataset),
        "val_size": len(val_dataset)
        }
    )

    train_lb_loader = DataLoader(
        dataset = train_lb_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = num_workers
    )
    train_ul_loader = DataLoader(
        dataset = train_ul_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = num_workers
    )
    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = num_workers
    )

    teacher = ViT(
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
    student = ViT(
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
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        teacher = torch.nn.DataParallel(teacher)
        student = torch.nn.DataParallel(student)

    teacher.to(args.device)
    student.to(args.device)
    teacher_optim = torch.optim.Adam(teacher.parameters(), lr=learning_rate)
    student_optim = torch.optim.Adam(student.parameters(), lr=learning_rate)
    # _, _, classes_weights = train_lb_dataset.get_info()

    teacher_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        teacher_optim, 
        T_max=args.epochs, 
        eta_min=1e-4
    )
    student_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        student_optim,
        T_max=args.epochs,
        eta_min=1e-4
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