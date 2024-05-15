from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from ultralytics import YOLO
from torchvision.transforms.functional import to_tensor
from PIL import Image
import torchvision
import cv2 as cv
from matplotlib import pyplot as plt
import torch


if __name__ == "__main__":
    person = 0
    truck = 7
    car = 2
    model = YOLO("yolov8x.pt", verbose = False)
    fpath = "/home/alberto/datasets/dreyeve-source/02_652.jpg"
    pred = model.predict(source = fpath)
    img = cv.imread(fpath)

    for b, p in zip(pred[0].boxes.xyxy, pred[0].boxes.conf):
        if p > 0.7:
            if (b[1] > 730 and b[2] - b[0] > 1200):
                cv.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2)
            else:
                cv.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 2)

    model2 = retinanet_resnet50_fpn_v2()
    torch_img = to_tensor(Image.open(fpath))
    model2.eval()
    pred = model2([torch_img])

    for b, l, p in zip(pred[0]["boxes"], pred[0]["labels"], pred[0]["scores"]):
        if p > 0.5 and l == person:
            cv.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2)

    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show() 
