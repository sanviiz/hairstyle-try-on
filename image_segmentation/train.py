import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import cv2
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = 5
BATCH_SIZE = 32
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 128  # 1024 originally
IMAGE_WIDTH = 128  # 1024 originally
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_FREQ = 2
IMG_DIR = 'CelebAMask-HQ\CelebA-HQ-img' # 'CelebAMask-HQ\debug_image\ori' # "CelebAMask-HQ\CelebA-HQ-img"
MASK_DIR = 'CelebAMask-HQ\CelebAMask-HQ-mask-anno\multiclass_dataset' # 'CelebAMask-HQ\debug_image\mask' # "CelebAMask-HQ\CelebAMask-HQ-mask-anno\multiclass_dataset"
SAVE_SAMPLE_IMAGE_FOLDER = "image_segmentation\saved_images"
SAVE_CHECKPOINTS_PATH ='image_segmentation\my_checkpoint_3.pth.tar'


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE, dtype=torch.int64) # targets.float().unsqueeze(1).to(device=DEVICE, dtype=torch.int64)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=1.0),
            A.RandomBrightnessContrast(p=0.2),
            # A.VerticalFlip(p=0.05),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=CLASSES).to(DEVICE).train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        IMG_DIR,
        MASK_DIR,
        BATCH_SIZE,
        transform,
        NUM_WORKERS,
        PIN_MEMORY,
        train_test_split=0.9
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("image_segmentation\my_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print('Epoch : ', epoch+1)
        print('====================================')
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }

        if (epoch + 1) % SAVE_FREQ == 0:
            save_checkpoint(checkpoint, filename=SAVE_CHECKPOINTS_PATH)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder=SAVE_SAMPLE_IMAGE_FOLDER, epoch_name='epoch_'+str(epoch+1), device=DEVICE
        )


if __name__ == "__main__":
    print(DEVICE)
    main()