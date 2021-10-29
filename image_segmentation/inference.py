import torch
import torchvision
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
sys.path.append(".")
from model import UNET
from utils import load_checkpoint, label_to_image
from dataset import celebamask_Dataset


def inference(image_path, model, transform, directory='inference_images'):
    model.eval()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmentations = transform(image=image)
    image = augmentations['image'].unsqueeze(dim=0).to(DEVICE)
    print(image.shape)
    with torch.no_grad():
        preds = model(image)
        preds = torch.nn.functional.softmax(preds, dim=1)
        preds = torch.argmax(preds, dim=1).float()
    preds = label_to_image(preds, label)
    preds = preds.permute(0, 3, 1, 2).to('cpu')

    if not os.path.exists(directory):
        os.makedirs(directory)

    torchvision.utils.save_image(
        preds[0], os.path.join(directory, 'test'+'.png')
    )

if __name__ == '__main__':
    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CLASSES = 5
    label = {
                'class':['others', 'neck_and_cloth','hair', 'skin', 'ear'],
                'color':[np.array([0,0,0]), np.array([0,255,0]), np.array([255,0,0]), np.array([0,0,255]), np.array([255,0,255])],
                'label': [0, 1, 2, 3, 4]
            }

    transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=CLASSES).to(DEVICE)
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    image_path = 'image_segmentation\\82.jpg'
    inference(image_path, model, transform, directory='inference_images')