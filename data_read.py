import random
import numpy as np
import os
import cv2
from os import scandir


def data_reader(input_dir, img_type='.jpeg'):
    """Read images from input_dir then shuffle them
  Args:
    input_dir: string, path of input dir, e.g., /path/to/dir
    img_type:
  Returns:
    file_paths: list of strings
  """
    file_paths = []
    file_labels = []
    label = -1
    for img_fold in scandir(input_dir):
        label = label + 1
        for img_file in scandir(img_fold):
            if img_file.name.endswith(img_type) and img_file.is_file():
                file_paths.append(img_file.path)
                file_labels.append(label)
    return file_paths, file_labels


def get_source_batch(batch_size, image_width, image_height, img_type='.jpeg', source_dir=""):
    file_paths, file_labels = data_reader(source_dir, img_type)
    maxSize = len(file_paths)
    if batch_size > maxSize:
        batch_size = maxSize
    idx_list = random.sample(range(0, maxSize), batch_size)
    files = []
    labels = []
    images = []
    if batch_size == 0:
        batch_size = maxSize
        for i in range(batch_size):
            files.append(file_paths[i])
            labels.append(file_labels[i])
            image = cv2.imread(os.path.join(file_paths[i]))
            image = cv2.resize(image, (image_width, image_height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image) / 127.5 - 1.
            images.append(image)
    else:
        for i in range(batch_size):
            for j in range(maxSize):
                if idx_list[i] == j:
                    files.append(file_paths[j])
                    labels.append(file_labels[j])
                    image = cv2.imread(os.path.join(file_paths[j]))
                    image = cv2.resize(image, (image_width, image_height))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = np.array(image) / 127.5 - 1.
                    images.append(image)
    return images, idx_list, len(file_paths), labels


def get_target_batch(batch_size, image_width, image_height, img_type='.jpeg', target_dir=""):
    file_paths, file_labels = data_reader(target_dir, img_type)
    maxSize = len(file_paths)
    if batch_size > maxSize:
        batch_size = maxSize - 1
    idx_list = random.sample(range(0, maxSize), batch_size)
    files = []
    labels = []
    images = []
    if batch_size == 0:
        batch_size = maxSize
        for i in range(batch_size):
            files.append(file_paths[i])
            labels.append(file_labels[i])
            image = cv2.imread(os.path.join(file_paths[i]))
            image = cv2.resize(image, (image_width, image_height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image) / 127.5 - 1.
            images.append(image)
    else:
        for i in range(batch_size):
            for j in range(maxSize):
                if idx_list[i] == j:
                    files.append(file_paths[j])
                    labels.append(file_labels[j])
                    image = cv2.imread(os.path.join(file_paths[j]))
                    image = cv2.resize(image, (image_width, image_height))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = np.array(image) / 127.5 - 1.
                    images.append(image)
    return images, idx_list, len(file_paths), labels


if __name__ == '__main__':
    # file_paths,file_labels=data_reader('/home/hit/paraProject/liaijia/cycle_datasets/source_banana', img_type='.jpg')
    # source_paths = get_source_batch(10, 224, 224, '.jpg')
    target_paths = get_target_batch(0, 224, 224, img_type='.png', target_dir="/home/hit/liaijia/FCGAN/datasets/malaria/test")
    # print(source_paths)
    print(target_paths)
    # print(data_reader('/home/hit/paraProject/liaijia/cycle_datasets/source_banana', img_type='.jpg'))
