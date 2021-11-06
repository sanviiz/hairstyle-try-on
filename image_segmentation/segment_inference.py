import torch
import torchvision
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from image_segmentation.model import UNET
from image_segmentation.utils import load_checkpoint, label_to_image
import yaml

def read_image_for_segment(image_path, transform, device='cuda'):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmentations = transform(image=image)
    image = augmentations['image'].unsqueeze(dim=0).to(device)
    return image
    
def predict(image, model, label):
    with torch.no_grad():
        pred = model(image)
        pred = torch.nn.functional.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1).float()
    pred = label_to_image(pred, label)
    pred = pred.permute(0, 3, 1, 2).to('cpu')
    return pred

class face_segment:
    def __init__(self, args, device=None):
        self.args = args
        with open(self.args.label_config, 'r') as f:
            self.label = yaml.safe_load(f)
            self.label['color'] = [np.array(c) for c in self.label['color']]   

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.transform = A.Compose(
                        [
                            A.Resize(height=self.args.image_height, width=self.args.image_width),
                            A.Normalize(
                                mean=[0.0, 0.0, 0.0],
                                std=[1.0, 1.0, 1.0],
                                max_pixel_value=255.0,
                            ),
                            ToTensorV2(),
                        ],
                        )
        self.model = UNET(in_channels=3, out_channels=5).to(device)
        load_checkpoint(torch.load(os.path.join(self.args.seg_model_path)), self.model)

    def segmenting(self, image_path):
        image = read_image_for_segment(image_path, self.transform, self.device)
        segmented_image = predict(image, self.model, self.label)

        # if not os.path.exists(self.arg.save_forder):
        #     os.makedirs(self.arg.save_forder)
        
        # torchvision.utils.save_image(
        # preds[0], os.path.join(self.arg.save_forder, 'test'+'.png')
        # )
        return segmented_image