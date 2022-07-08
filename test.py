from image_segmentation.segment_inference import face_segment
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse

images_path = glob.glob('report_images\\test_image\\*.jpg')

def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--seg_model_path', type=str, default=os.path.join("image_segmentation", "my_checkpoint.pth.tar"), help='Path to the segmentation model')
    parser.add_argument('--image_size', type=tuple, default=(256,256), help='output image size (height, width)')
    parser.add_argument('--input_image_size', type=tuple, default=(256,256), help='input image size before segment (height, width)')
    parser.add_argument('--label_config', type=str, default=os.path.join("image_segmentation", "label.yml"), help='Path to the label.yml')
    parser.add_argument('--save_forder', type=str, default='segmented_images', help='Path to the segmented folder')

    args = parser.parse_args()
    return args

args = parse_args()
segment = face_segment(args)
segmented_images = []

for img_path in images_path:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    segmented_img = segment.segmenting(img)
    segmented_images.append(segmented_img)
    cv2.imwrite(os.path.join('report_images/test_image/segmented', 'segmented_'+img_path.split('\\')[-1]), cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR))


