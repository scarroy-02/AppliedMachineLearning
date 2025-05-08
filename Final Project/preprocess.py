import os
import time
import pickle
import numpy as np
from PIL import Image
import pandas as pd

# === Compute frontal score ===
def compute_frontal_score(row):
    lx, ly = row['lefteye_x'], row['lefteye_y']
    rx, ry = row['righteye_x'], row['righteye_y']
    nx, ny = row['nose_x'], row['nose_y']
    lmx, lmy = row['leftmouth_x'], row['leftmouth_y']
    rmx, rmy = row['rightmouth_x'], row['rightmouth_y']
    eye_symmetry = abs((rx - nx) - (nx - lx))
    mouth_symmetry = abs((rmx - nx) - (nx - lmx))
    eye_level_diff = abs(ly - ry)
    angle_rad = np.arctan2(ry - ly, rx - lx)
    eye_angle = abs(np.degrees(angle_rad))
    return eye_symmetry + mouth_symmetry + eye_level_diff + eye_angle

# === Get filtered indices ===
def get_filtered_indices(landmark_csv_path, threshold=15.0):
    df = pd.read_csv(landmark_csv_path)
    df.columns = ['image_name', 'lefteye_x', 'lefteye_y', 'righteye_x', 'righteye_y',
                  'nose_x', 'nose_y', 'leftmouth_x', 'leftmouth_y', 'rightmouth_x', 'rightmouth_y']
    df['frontal_score'] = df.apply(compute_frontal_score, axis=1)
    filtered = df[df['frontal_score'] < threshold]
    filtered['index'] = filtered['image_name'].str.replace('.jpg', '').astype(int)
    return filtered['index'].tolist()

# === Read landmarks from CSV ===
def read_landmarks(landmark_csv_path):
    df = pd.read_csv(landmark_csv_path)
    df.columns = ['image_name', 'lefteye_x', 'lefteye_y', 'righteye_x', 'righteye_y',
                  'nose_x', 'nose_y', 'leftmouth_x', 'leftmouth_y', 'rightmouth_x', 'rightmouth_y']
    landmarks = df[['lefteye_x', 'lefteye_y', 'righteye_x', 'righteye_y',
                    'nose_x', 'nose_y', 'leftmouth_x', 'leftmouth_y', 'rightmouth_x', 'rightmouth_y']].values
    return landmarks

# === Read image ===
def read_one_image(image_folder_path, image_name):
    image_path = os.path.join(image_folder_path, image_name)
    image = Image.open(image_path)
    return image

# === Resize and crop ===
def resize_one_image(index, image, landmarks, save_flag=False, save_folder_path='./celeba_data/image_align_processed'):
    image_data = np.asarray(image, dtype="int32")
    height, width = image_data.shape[:2]
    left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y, left_mouth_x, left_mouth_y, right_mouth_x, right_mouth_y = landmarks[index-1]
    length_crop = 80
    edge_to_eye = 20
    edge_to_mouth = 60
    length_resize = 80
    if left_eye_x - edge_to_eye < 0:
        left = 0
        right = left + length_crop
    elif right_eye_x + edge_to_eye > width - 1:
        right = width - 1
        left = right - length_crop
    else:
        left = left_eye_x - edge_to_eye
        right = left + length_crop
    mouth_mean_y = np.mean([left_mouth_y, right_mouth_y])
    if mouth_mean_y - edge_to_mouth < 0:
        upper = 0
        lower = upper + length_crop
    elif mouth_mean_y + (length_crop - edge_to_mouth) > height - 1:
        lower = height - 1
        upper = lower - length_crop
    else:
        upper = mouth_mean_y - edge_to_mouth
        lower = upper + length_crop
    image_cropped = image.crop((left, upper, right, lower))
    image_resized = image_cropped.resize((length_resize, length_resize))
    if save_flag:
        resized_image_name = 'resized_' + str(length_resize) + '_' + str(index).zfill(6) + '.jpg'
        image_resized.save(os.path.join(save_folder_path, resized_image_name))
    return image_resized

# === Write only filtered images to pickle ===
def write_filtered_to_pickle(landmark_csv_path='./celeba_data/list_landmarks_align_celeba.csv',
                              image_folder_path='./celeba_data/image_align',
                              save_folder_path='./celeba_data/pickle',
                              filtered_indices=[],
                              verbose_step=1000,
                              save_step=50000):
    image_data_list = []
    landmarks = read_landmarks(landmark_csv_path)
    print('Landmarks read')

    os.makedirs(save_folder_path, exist_ok=True)

    last_saved_index = 0
    tic = time.time()
    for count, index in enumerate(filtered_indices, 1):
        image_name = str(index).zfill(6) + '.jpg'
        image = read_one_image(image_folder_path, image_name)
        image_resized = resize_one_image(index, image, landmarks)
        image_resized_data = np.asarray(image_resized, dtype='uint8')
        image_data_list.append(image_resized_data)

        if count % verbose_step == 0:
            print('Completed image index:', index)

        if count % save_step == 0 or count == len(filtered_indices):
            image_data_array = np.array(image_data_list)
            pickle_name = str(last_saved_index + 1).zfill(6) + '_' + str(index).zfill(6) + '.pickle'
            pickle_path = os.path.join(save_folder_path, pickle_name)
            pickle.dump(image_data_array, open(pickle_path, 'wb'))
            toc = time.time()
            print('Saved to file:', pickle_name)
            print('Time for this batch:', toc - tic, 's')
            image_data_list = []
            last_saved_index = index
            tic = time.time()

# === Run ===
landmark_csv_path = './celeba_data/list_landmarks_align_celeba.csv'  # your renamed CSV
filtered_indices = get_filtered_indices(landmark_csv_path, threshold=15.0)
write_filtered_to_pickle(filtered_indices=filtered_indices)
