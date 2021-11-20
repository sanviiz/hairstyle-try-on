import cv2
import numpy as np

# ============= Call this to use =============
# from artifact_fill import face_artifact_fill
#
# Parameter
# face_artifact_fill(original_image, original_mask, new_image,
#                        new_mask, new_segment):


# define blue channel color
blue_channel = 2    # for RGB
# blue_channel = 0    # for BGR


def convert_to_gray(image):
    return cv2.cvtColor(image.astype('uint8'), cv2.COLOR_RGB2GRAY)


def background_artifact_fill(original_image, original_mask, new_image, background_radius):
    original_mask_gray = cv2.cvtColor(original_mask,
                                      cv2.COLOR_RGB2GRAY)
    object_mask = np.where(original_mask_gray != 0, 255, 0)
    object_mask = object_mask.astype('uint8')
    dilation_kernel = np.ones((13, 13), np.uint8)
    dilation_mask = cv2.dilate(object_mask, dilation_kernel, iterations=1)
    filled_background = cv2.inpaint(original_image, dilation_mask,
                                    background_radius, cv2.INPAINT_TELEA)
    filled_background = np.where(new_image == np.array([255, 255, 255]),
                                 filled_background, new_image)

    return filled_background


def face_artifact_fill(original_image, original_mask, new_image,
                       new_mask, new_segment, face_radius=5, background_radius=25,
                       had_background=0):
    if had_background:
        filled_bg_artifact = background_artifact_fill(
            original_image, original_mask, new_image, background_radius).astype('uint8')
    else:
        filled_bg_artifact = new_image.astype('uint8')
    diff = np.where((convert_to_gray(new_mask) == convert_to_gray(new_segment)),
                    0, 255).astype('uint8')
    filled_face_artifact = cv2.inpaint(filled_bg_artifact, diff,
                                       face_radius, cv2.INPAINT_TELEA)

    return filled_face_artifact

