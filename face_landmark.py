import cv2
import mediapipe as mp
import os
import numpy as np
import matplotlib.pyplot as plt

# INITIALIZING OBJECTS
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# cap = cv2.VideoCapture(0)
IMAGE_FOLDER = 'CelebAMask-HQ\CelebA-HQ-img'
target_image_name = '17.jpg'
source_image_name = '100.jpg'


def get_xy_coor(results, image_size):
    x_coors = []
    y_coors = []
    point_lm_index = [10, 152, 234]  # [10, 152, 234, 454]
    for face in results.multi_face_landmarks:
        for landmark in face.landmark:
            x = landmark.x
            y = landmark.y

            shape = image_size
            relative_x = int(x * shape[1])
            relative_y = int(y * shape[0])
            x_coors.append(relative_x)
            y_coors.append(relative_y)
    return np.array(x_coors)[point_lm_index], np.array(y_coors)[point_lm_index]


with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
    target_image = cv2.imread(os.path.join(IMAGE_FOLDER, target_image_name))
    source_image = cv2.imread(os.path.join(IMAGE_FOLDER, source_image_name))

    target_mask_image = cv2.imread(
        os.path.join(IMAGE_FOLDER, target_image_mask_name))
    source_mask_image = cv2.imread(
        os.path.join(IMAGE_FOLDER, source_image_mask_name))

    # Convert the BGR image to RGB before processing.
    target_results = face_mesh.process(
        cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
    source_results = face_mesh.process(
        cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    # if not results.multi_face_landmarks:

    # annotated_image = image.copy()

    tx, ty = get_xy_coor(target_results, target_image.shape)
    sx, sy = get_xy_coor(source_results, source_image.shape)

    t_coors = []
    s_coors = []
    for idx in range(len(tx)):
        t_coors.append((tx[idx], ty[idx]))
        s_coors.append((sx[idx], sy[idx]))

    t_coors = np.array(t_coors).astype(np.float32)
    s_coors = np.array(s_coors).astype(np.float32)
    diff = (t_coors - s_coors)[0]
    print(diff)
    # create the translation matrix using diff, it is a NumPy array

    # translation_matrix = np.array(
    #     [[1, 0, diff[0]], [0, 1, diff[1]]], dtype=np.float32)

    translation_matrix = cv2.getAffineTransform(s_coors, t_coors)

    print(t_coors, s_coors)
    print(translation_matrix)

    translated_image = cv2.warpAffine(src=source_image, M=translation_matrix, dsize=(
        source_image.shape[1], source_image.shape[0]))

    translated_mask = cv2. warpAffine(src=source_mask_image, M=translation_matrix, dsize=(
        source_mask_image.shape[1], source_mask_image.shape[0]))
    print(translated_mask.shape)

    hair_map = np.zeros((translated_mask.shape[:2]), dtype=np.int)
    hair_map[(translated_mask == np.array([0, 0, 255])).all(axis=2)] = 255
    print(hair_map.shape)

    t = translated_image[:, :, 0].copy()
    translated_image = np.where(hair_map != 255, 0, )

    # croped_image = np.zeros((translated_image.shape[:2]), dtype=np.int)
    t[(hair_map != 0).all()] = 0

    plt.imshow(translated_image)
    # plt.figure()
    # plt.imshow(translated_image)
    # plt.figure()
    # plt.imshow(translated_mask)
    # plt.figure()
    # plt.imshow(target_image)

    plt.show()

    # cv2.imshow('Source image', source_image)
    # cv2.imshow('translated image', translated_image)
    # cv2.imshow('target image', target_image)

    # cv2.waitKey()