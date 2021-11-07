import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import cv2
import argparse
import os
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# # Hyperparameters etc.
# LEARNING_RATE = 1e-4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CLASSES = 5
# BATCH_SIZE = 32
# NUM_EPOCHS = 50
# NUM_WORKERS = 2
# IMAGE_HEIGHT = 128  # 1024 originally
# IMAGE_WIDTH = 128  # 1024 originally
# PIN_MEMORY = True
# LOAD_MODEL = False
# SAVE_FREQ = 2
# IMG_DIR = 'CelebAMask-HQ\CelebA-HQ-img' # 'CelebAMask-HQ\debug_image\ori' # "CelebAMask-HQ\CelebA-HQ-img"
# MASK_DIR = 'CelebAMask-HQ\CelebAMask-HQ-mask-anno\multiclass_dataset' # 'CelebAMask-HQ\debug_image\mask' # "CelebAMask-HQ\CelebAMask-HQ-mask-anno\multiclass_dataset"
# SAVE_SAMPLE_IMAGE_FOLDER = "image_segmentation\saved_images"
# SAVE_CHECKPOINTS_PATH ='image_segmentation\my_checkpoint_3.pth.tar'


def train_unet_parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='device')
    parser.add_argument('--classes', type=int, default=5, help='number of classes')
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epoch')
    parser.add_argument('--image_height', type=int, default=256, help='output image height')
    parser.add_argument('--image_width', type=int, default=256, help='output image width')
    parser.add_argument('--pin_memory', type=bool, default=True, help='pin_memory')
    parser.add_argument('--load_model', type=bool, default=False, help='load exist model')
    parser.add_argument('--model_checkpoint_path', type=str, default='', help='if load_model load from this path')
    parser.add_argument('--save_model_freq', type=int, default=1, help='save model every ... epoch')
    parser.add_argument('--image_dir', type=str, default=os.path.join("CelebAMask-HQ", "CelebA-HQ-img"), help='path to image directory')
    parser.add_argument('--mask_dir', type=str, default=os.path.join("CelebAMask-HQ", "CelebAMask-HQ-mask-anno", "multiclass_dataset"), help='path to mask directory')
    parser.add_argument('--save_sample_image_dir', type=str, default=os.path.join("image_segmentation", "saved_images"), help='path to save sample image')
    parser.add_argument('--save_model_checkpoint_path', type=str, default=os.path.join("image_segmentation", "my_checkpoint.pth.tar"), help='path to new model checkpoint')
    parser.add_argument('--train_test_split_size', type=float, default=0.9, help='train size')
    parser.add_argument('--save_sample_image_freq', type=int, default=1, help='save sample image each ... epoch')

    args = parser.parse_args()
    return args


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=args.device)
        targets = targets.to(device=args.device, dtype=torch.int64) # targets.float().unsqueeze(1).to(device=DEVICE, dtype=torch.int64)

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
            A.Resize(height=args.image_height, width=args.image_width),
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

    model = UNET(in_channels=3, out_channels=args.classes).to(args.device).train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader, val_loader = get_loaders(
        args.image_dir,
        args.mask_dir,
        args.batch_size,
        transform,
        args.num_workers,
        args.pin_memory,
        args.train_test_split_size
    )

    if args.load_model:
        load_checkpoint(torch.load(args.model_checkpoint_path), model)


    check_accuracy(val_loader, model, device=args.device)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.n_epochs):
        print('Epoch : ', epoch+1)
        print('====================================')
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }

        if (epoch + 1) % args.save_model_freq == 0:
            save_checkpoint(checkpoint, filename=args.save_model_checkpoint_path)

        # check accuracy
        check_accuracy(val_loader, model, device=args.device)

        # print some examples to a folder
        if (epoch + 1) % args.save_sample_image_freq == 0:
            save_predictions_as_imgs(
                val_loader, model, folder=args.save_sample_image_dir, epoch_name='epoch_'+str(epoch+1), device=args.device
            )


if __name__ == "__main__":
    args = train_unet_parse_args()
    print(args.device)
    main()