from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead, sigmoid_focal_loss
from functools import partial
import torch
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--n_iters", type=int, default=1000)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.001)
args = parser.parse_args()
args.device = torch.device(args.device)


def train_loop(
    model,
    optimizer,
    scheduler,
    train_loader,
    test_loader,
    args
) -> None:

    pbar = tqdm.tqdm(total=args.n_iters, desc="Training")
    train_iter = iter(train_loader)
    try:
        batch = next(train_iter)
    
    except:
        train_iter = iter(train_loader)
        batch = next(train_iter)

    for step in range(args.n_iters):
        images, targets = batch
        images, targets = images.to(args.device), targets.to(args.device)
        optimizer.zero_grad()
        preds, losses = model(images, targets)
        exit()
        
        
    


def create_model(n_classes):
    model = retinanet_resnet50_fpn_v2(weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=n_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32)
    )
    return model


if __name__ == "__main__":
    classes = ["background", "car", "person"]
    classes_id = [i for i in range(len(classes))]
    model = create_model(n_classes = len(classes))
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_iters)
    
    train_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=None,
        test_loader=None,
        args=args
    )

    


from nuimages import NuImages
import matplotlib.pyplot as plt
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights


nuim = NuImages(dataroot='./data/nuscenes/', version='v1.0-mini', verbose=True, lazy=True)
sample = nuim.sample[0]
object_tokens, surface_tokens = nuim.list_anns(sample['token'])
key_camera_token = sample['key_camera_token']

# nuim.render_image(key_camera_token, annotation_type='none',
#                   out_path = './data/tmp.png')

model = retinanet_resnet50_fpn_v2(weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
print(object_tokens)