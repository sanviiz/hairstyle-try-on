import torch
import torchvision
from dataset import celebamask_Dataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

label = {
            'class':['others', 'neck_and_cloth','hair', 'skin', 'ear'],
            'color':[np.array([0,0,0]), np.array([0,255,0]), np.array([255,0,0]), np.array([0,0,255]), np.array([255,0,255])],
            'label': [0, 1, 2, 3, 4]
        }

def get_loaders(train_dir, train_maskdir, batch_size, train_transform, num_workers, pin_memory=True, train_test_split=0.9):
    ds = celebamask_Dataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_size = int(len(ds) * 0.9)
    val_size = len(ds) - train_size

    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x) 
            preds = torch.nn.functional.softmax(preds, dim=1)
            preds = torch.argmax(preds, dim=1).float() # (preds > 0.5).float()

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    model.train()
    return num_correct/num_pixels

def fill_color(mask3darray, label):
    for idx, label_number in enumerate(label['label']):
        t = np.where(mask3darray[:, :, 0]==label_number, True, False)
        # red channal
        mask3darray[:, :, 0][t] = label['color'][idx][0]
        # green channal
        mask3darray[:, :, 1][t] = label['color'][idx][1]
        # blue channal
        mask3darray[:, :, 2][t] = label['color'][idx][2]

    return mask3darray

def label_to_image(preds, label):
    masks = []
    for idx in range(preds.shape[0]): # batch size = preds.shape[0]
        mask = np.asarray([preds[idx].cpu().numpy()]*3)
        mask = mask.transpose(1, 2, 0)

        mask = fill_color(mask, label)
        # print("=====================\n")
        # print(mask[:,:,0], mask[:,:,1], mask[:,:,2])
        masks.append(mask)
    return torch.from_numpy(np.array(masks))

def save_predictions_as_imgs(loader, model, folder="saved_images", epoch_name='', device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
            preds = torch.nn.functional.softmax(preds, dim=1)
            preds = torch.argmax(preds, dim=1).float() # (preds > 0.5).float()
        preds = label_to_image(preds, label)
        preds = preds.permute(0, 3, 1, 2).to('cpu')
        # print(preds.shape)

        directory = os.path.join(folder, epoch_name) # f"{folder}/{epoch}/pred_{idx}.png"
        if not os.path.exists(directory):
            os.makedirs(directory)

        torchvision.utils.save_image(
            preds[0], os.path.join(directory, 'pred_'+str(idx)+'.png')
        )
        # torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()

if __name__ == '__main__':
    preds = [[1,2,1], [0,0,1], [1,2,2]]
    with open('test_numpy_image.npy', 'rb') as f:
        preds = np.load(f)
    print(preds.max())

    preds = torch.from_numpy(np.array(preds)).type(torch.float32)
    preds = preds.unsqueeze(axis=0)
    label_to_image(preds, label)