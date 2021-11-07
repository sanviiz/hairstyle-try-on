import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_segmentation.segment_inference import face_segment
from runners.image_editing import Diffusion
from image_landmark_transform.face_landmark import face_landmark_transform


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    # SDEdit
    parser.add_argument('--config', type=str, default=os.path.join("configs", "celeba.yml"), help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--sample', action='store_true', help='Whether to produce samples from the model')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    # parser.add_argument('--npy_name', type=str, required=True)
    parser.add_argument('--sample_step', type=int, default=3, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=400, help='Sampling noise scale')

    # image segmentation
    parser.add_argument('--seg_model_path', type=str, default=os.path.join("image_segmentation", "my_checkpoint.pth.tar"), help='Path to the segmentation model')
    parser.add_argument('--target_image_path', type=str, required=True, help='Path to the target image path')
    parser.add_argument('--source_image_path', type=str, required=True, help='Path to the source image path')
    parser.add_argument('--image_size', type=tuple, default=(256,256), help='output image size (height, width)')
    parser.add_argument('--input_image_size', type=tuple, default=(128,128), help='input image size before segment (height, width)')
    parser.add_argument('--label_config', type=str, default=os.path.join("image_segmentation", "label.yml"), help='Path to the label.yml')
    parser.add_argument('--save_forder', type=str, default='segmented_images', help='Path to the segmented folder')

    args = parser.parse_args()

    # parse config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(os.path.join(args.exp, 'image_samples'), exist_ok=True)
    args.image_folder = os.path.join(args.exp, 'image_samples', args.image_folder)
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)  
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input("Image folder already exists. Overwrite? (Y/N)")
            if response.upper() == 'Y':
                overwrite = True
        
        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)
    
    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    torch.backends.cudnn.benchmark = True

    return args, new_config

def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def resize_image(image, image_size:tuple):
    image_height, image_width = image_size[0], image_size[1]
    return cv2.resize(image, (image_height, image_width), interpolation = cv2.INTER_NEAREST)

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def main():
    args, config = parse_args_and_config()
    segment = face_segment(args)
    # read original image
    target_image = read_image(args.target_image_path)
    source_image = read_image(args.source_image_path)

    # infer image segmentation
    target_mask = segment.segmenting(image=target_image)
    source_mask = segment.segmenting(image=source_image)

    # resize image and mask
    target_image = resize_image(target_image, args.image_size)
    source_image = resize_image(source_image, args.image_size)
    target_mask = resize_image(target_mask, args.image_size)
    source_mask = resize_image(source_mask, args.image_size)


    # detect face landmark and transform image
    face_landmark_transform(target_image, target_mask, source_image, source_mask)

    plt.imshow(target_image)
    plt.figure()
    plt.imshow(source_image)
    plt.figure()
    plt.imshow(target_mask)
    plt.figure()
    plt.imshow(source_mask)
    plt.show()

if __name__ == '__main__':
    sys.exit(main())
