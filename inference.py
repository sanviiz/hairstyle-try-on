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
    parser.add_argument('--image_height', type=int, default=128, help='segmented image height')
    parser.add_argument('--image_width', type=int, default=128, help='segmented image width')
    parser.add_argument('--label_config', type=str, default=os.path.join("image_segmentation", "label.yml"), help='Path to the label.yml')
    parser.add_argument('--save_forder', type=str, default='segmented_images', help='Path to the segmented folder')

    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    return args

def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_config(file):
    with open(os.path.join('tutorial_code', 'configs', file), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    return new_config

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
    args = parse_args_and_config()
    segment = face_segment(args)
    target_image = read_image(args.target_image_path)
    source_image = read_image(args.source_image_path)
    target_segment = segment.segmenting(image_path=args.target_image_path)
    source_segment = segment.segmenting(image_path=args.source_image_path)

    plt.imshow(target_image)
    plt.figure()
    plt.imshow(source_image)
    plt.figure()
    plt.imshow(target_segment.squeeze().permute(1, 2, 0))
    plt.figure()
    plt.imshow(source_segment.squeeze().permute(1, 2, 0))
    plt.show()

if __name__ == '__main__':
    sys.exit(main())
