import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============= Call this to use =============
# from artifact_fill import face_artifact_fill


## define blue channel color 
blue_channel = 2    # for RGB
#blue_channel = 0    # for BGR


def background_artifact_fill(original_image, original_mask, new_hair_image):
    original_mask_gray = cv2.cvtColor(original_mask, cv2.COLOR_RGB2GRAY)
    object_mask = np.where(original_mask_gray != 0, 255, 0)
    object_mask = object_mask.astype('uint8')

    dilation_kernel = np.ones((13, 13), np.uint8)
    dilation_mask = cv2.dilate(object_mask, dilation_kernel, iterations=1)
    filled_background = cv2.inpaint(
        original_image, dilation_mask, 25, cv2.INPAINT_TELEA)

    filled_background = np.where(new_hair_image == np.array(
        [255, 255, 255]), filled_background, new_hair_image)

    return filled_background


def face_artifact_fill(original_image, original_mask, new_hair_image, new_mask):
    artifact_face_mask = new_mask[:, :, blue_channel].copy()
    artifact_face_mask = (
        artifact_face_mask - original_mask[:, :, blue_channel]
    )

    filled_bg_artifact = background_artifact_fill(
        original_image, original_mask, new_hair_image)

    filled_face_artifact = cv2.inpaint(
        filled_bg_artifact.astype('uint8'), artifact_face_mask.astype('uint8'), 5, cv2.INPAINT_TELEA
    )

    return filled_face_artifact, artifact_face_mask, filled_bg_artifact

