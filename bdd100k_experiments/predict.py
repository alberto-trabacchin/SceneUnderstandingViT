import argparse
import data
import torch
from vit_pytorch import SimpleViT
from torchvision.utils import save_image
from pathlib import Path
import shutil
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./bdd_1k')
    parser.add_argument('--model-path', type=str, default='./models')
    parser.add_argument('--img-size', type=int, required=True)
    return parser.parse_args()


def save_predictions(model, test_dataset, class2idx):
    pred_path = Path(f"{args.data_path}/predictions")
    if pred_path.exists():
        shutil.rmtree(pred_path)
    (pred_path / "safe").mkdir(parents=True, exist_ok=True)
    (pred_path / "dangerous").mkdir(parents=True, exist_ok=True)
    model.eval()
    print("Saving predictions...")
    pbar = tqdm(total=len(test_dataset), position=0, leave=True)
    
    for i in range(len(test_dataset)):
        img, _ = test_dataset[i]
        img = img.unsqueeze(0)
        pred = model(img)
        pred = torch.argmax(pred, dim=1)
        if pred == class2idx["safe"]:
            save_image(img, pred_path / "safe" / f"{i}.png")
        elif pred == class2idx["dangerous"]:
            save_image(img, pred_path / "dangerous" / f"{i}.png")
        else:
            raise ValueError("Invalid prediction")
        pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    # Note: dangerous = 0, safe = 1
    args = parse_args()
    model = SimpleViT ( 
        image_size = args.img_size,
        patch_size = 20,
        num_classes = 2,
        dim = 512,
        depth = 4,
        heads = 4,
        mlp_dim = 128
    )
    _, val_dataset, test_dataset, _ = data.get_bdd100k(args)
    class2idx = val_dataset.class_to_idx
    with open(f"{args.model_path}", "rb") as f:
        model.load_state_dict(torch.load(f))
    save_predictions(model, test_dataset, class2idx)