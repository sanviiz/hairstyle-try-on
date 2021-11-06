import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import yaml



class celebamask_Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        with open(os.path.join("image_segmentation", "label.yml"), 'r') as f:
            self.label = yaml.safe_load(f)
            self.label['color'] = [np.array(c) for c in self.label['color']]    

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png"))
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path) # np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # mask[mask == 255.0] = 1.0
        # mask = mask // 255.
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            
        mask_label = self.image_to_label(mask, self.label)

        # mask_label = np.zeros((mask.shape[:2]), dtype=np.int)
        # mask_label[(mask.numpy()==self.label['color'][0]).all(axis=2)] = 0
        # mask_label[(mask.numpy()==self.label['color'][1]).all(axis=2)] = 1
        # mask_label[(mask.numpy()==self.label['color'][2]).all(axis=2)] = 2
        # mask_label[(mask.numpy()==self.label['color'][3]).all(axis=2)] = 3
        # mask_label[(mask.numpy()==self.label['color'][4]).all(axis=2)] = 4

        mask_label = torch.from_numpy(mask_label)
        mask_label = mask_label.type(torch.LongTensor)
        
        # plt.imshow(mask_label)
        # plt.show()
        
        return image, mask_label
    
    def image_to_label(self, mask, label):
        mask_label = np.zeros((mask.shape[:2]), dtype=np.int)
        for idx, la in enumerate(label['label']):
            mask_label[(mask.numpy()==label['color'][idx]).all(axis=2)] = la
        return mask_label