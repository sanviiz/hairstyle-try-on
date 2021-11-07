import cv2
import mediapipe as mp
import os
import numpy as np
import matplotlib.pyplot as plt

# Initializing objects
mp_face_mesh = mp.solutions.face_mesh


def get_xy_coordinates(results, image_size):
    x_coordinates = []
    y_coordinates = []
    point_lm_index = [10, 152, 234]  # Face landmark position
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


def crop_image_by_mask(method, image, mask):
    if method == 'hair_only':
        generate_mask = 255 * \
            np.ones((mask.shape), dtype=int)
        generate_mask[(mask == np.array(
            [255, 0, 0])).all(axis=2)] = 0  # Red only
        croped_image = np.where(
            generate_mask == 0, image, 255)
        croped_mask = np.where(generate_mask == 0, mask, 0)
    elif method == "no_hair":
        generate_mask = np.zeros(
            (mask.shape), dtype=int)
        generate_mask[(mask == np.array(
            [255, 0, 0])).all(axis=2) | (mask == np.array(
                [0, 0, 0])).all(axis=2)] = 255  # Red and black
        croped_image = np.where(
            generate_mask == 0, image, 255)
        croped_mask = np.where(generate_mask == 0, mask, 0)
    return croped_image, croped_mask


def merge_head_part(type, hair, face):
    if type == 'image':
        return np.where(hair != 255, hair, face)
    elif type == 'mask':
        head = face.copy()
        head[(hair == np.array([255, 0, 0])).all(
            axis=2)] = np.array([255, 0, 0])
        return head


def face_landmark_transform(static_image, static_mask, transform_image, transform_mask):
    static_landmark_coordinates, transform_landmark_coordinates = get_landmark_coordinates(
        static_image), get_landmark_coordinates(
        transform_image)

    affine_matrix = cv2.getAffineTransform(
        transform_landmark_coordinates, static_landmark_coordinates)

    transformed_image, transformed_mask = transform_process(
        transform_image, affine_matrix), transform_process(
        transform_mask, affine_matrix)

    static_image_no_hair, static_mask_no_hair = crop_image_by_mask(
        'no_hair', static_image, static_mask)
    transformed_image_hair, transformed_mask_hair = crop_image_by_mask(
        'hair_only', transformed_image, transformed_mask)

    face_landmark_transform_image = merge_head_part(
        'image', transformed_image_hair, static_image_no_hair)
    face_landmark_transform_mask = merge_head_part(
        'mask', transformed_mask_hair, static_mask_no_hair)

    output_object = {
        'result_image': face_landmark_transform_image,
        'result_mask': face_landmark_transform_mask,
        'hair_mask': transformed_mask_hair,
        'no_hair_mask': static_mask_no_hair,
        'transformed_mask': transformed_mask,
        'transformed_image': transformed_image,
    }

    return output_object
