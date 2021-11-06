import cv2
import mediapipe as mp
import os
import numpy as np
import matplotlib.pyplot as plt

# INITIALIZING OBJECTS
mp_face_mesh = mp.solutions.face_mesh

IMAGE_FOLDER = 'images'

static_image_name = '82.jpg'
transform_image_name = '92.jpg'

static_image_mask_name = '82_seg.png'
transform_image_mask_name = '92_seg.png'

static_image = cv2.imread(os.path.join(
    IMAGE_FOLDER, static_image_name))
transform_image = cv2.imread(os.path.join(
    IMAGE_FOLDER, transform_image_name))

static_mask = cv2.imread(
    os.path.join(IMAGE_FOLDER, static_image_mask_name))
transform_mask = cv2.imread(
    os.path.join(IMAGE_FOLDER, transform_image_mask_name))


def resize_image(image):
    return cv2.resize(image, (256, 256),
                      interpolation=cv2.INTER_AREA)


# Resize image before processing.
static_image = resize_image(static_image)
static_mask = resize_image(static_mask)
transform_image = resize_image(transform_image)
transform_mask = resize_image(transform_mask)


def get_xy_coordinates(results, image_size):
    x_coordinates = []
    y_coordinates = []
    point_lm_index = [10, 152, 234]  # face landmark position
    for face in results.multi_face_landmarks:
        for landmark in face.landmark:
            x = landmark.x
            y = landmark.y
            shape = image_size
            relative_x = int(x * shape[1])
            relative_y = int(y * shape[0])
            x_coordinates.append(relative_x)
            y_coordinates.append(relative_y)
    return np.array(x_coordinates)[point_lm_index], np.array(y_coordinates)[point_lm_index]


def get_landmark_coordinates(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        landmark_results = face_mesh.process(image)
        x, y = get_xy_coordinates(landmark_results, image.shape)
        landmark_coordinates = []
        for idx in range(len(x)):
            landmark_coordinates.append((x[idx], y[idx]))
        landmark_coordinates = np.array(
            landmark_coordinates).astype(np.float32)
        return landmark_coordinates


def transform_process(image, matrix):
    return cv2.warpAffine(src=image, M=matrix, dsize=(
        image.shape[1], image.shape[0]))


def face_landmark_transform(static_image, static_mask, transform_image, transform_mask):
    static_landmark_coordinates, transform_landmark_coordinates = get_landmark_coordinates(
        static_image), get_landmark_coordinates(
        transform_image)

    affine_matrix = cv2.getAffineTransform(
        transform_landmark_coordinates, static_landmark_coordinates)

    transformed_image, transformed_mask = transform_process(
        transform_image, affine_matrix), transform_process(
        transform_mask, affine_matrix)

    static_mask_no_hair = np.zeros(
        (static_mask.shape), dtype=int)
    static_mask_no_hair[(static_mask == np.array(
        [0, 0, 255])).all(axis=2) | (static_mask == np.array(
            [0, 0, 0])).all(axis=2)] = 255
    static_image_no_hair = np.where(
        static_mask_no_hair == 0, static_image, 255)

    transformed_mask_hair = 255 * \
        np.ones((transformed_mask.shape), dtype=int)
    transformed_mask_hair[(transformed_mask == np.array(
        [0, 0, 255])).all(axis=2)] = 0
    transformed_image_hair = np.where(
        transformed_mask_hair == 0, transformed_image, 255)

    final_image = np.where(
        transformed_image_hair != 255, transformed_image_hair, static_image_no_hair)

    plt.imshow(final_image)
    plt.show()


if __name__ == "__main__":
    face_landmark_transform(static_image, static_mask,
                            transform_image, transform_mask)
