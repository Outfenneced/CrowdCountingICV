import sys
import time

import cv2
import os
import urllib.request
import zipfile
import random

import numpy as np
import scipy.io as sio
import skimage


IMAGE_FILE_FORMAT = "img_{:04}.jpg"
MAT_FILE_FORMAT = "img_{:04}_ann.mat"
WINDOW_SHAPE = 224


def generate_marks_map(marks, shape):
    marks = marks.astype(dtype=np.uint16)
    marks_orig_shape = marks.shape
    marks = marks[np.logical_and(marks[:, 0] < shape[1], marks[:, 1] < shape[0])]
    marks_new_shape = marks.shape
    if marks_orig_shape != marks_new_shape:
        print("Removed extraneous values, count: ", marks_orig_shape[0] - marks_new_shape[0])
    ys, xs = marks.T
    mapping = np.zeros(shape, dtype=np.uint8)
    mapping[xs, ys] = 1
    return mapping


def get_marks_from_file(mat_path, shape):
    mat = sio.loadmat(mat_path)['annPoints']
    marks = generate_marks_map(mat, shape)
    return marks


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def download_dataset(out_dir):
    out_path = os.path.join(out_dir, "dataset.zip")
    urllib.request.urlretrieve("http://crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_ECCV18.zip", out_path, reporthook=reporthook)
    with zipfile.ZipFile(out_path, "r") as zip_ref:
        zip_ref.extractall("dir")
    os.remove(out_path)


def preprocess(data_dir, output_dir, remove=False):
    random.seed = 1
    subdirs = ["Train", "Test"]
    for subdir in subdirs:
        file_count = 0
        full_dir = os.path.join(data_dir, subdir)
        out_dir = os.path.join(output_dir, subdir)
        os.makedirs(out_dir, exist_ok=True)
        files = os.listdir(full_dir)
        max_num = len(files) // 2
        image_nums = list(range(1, max_num+1))
        if subdir == "Train":
            random.shuffle(image_nums)
            image_nums = image_nums[:len(image_nums)//2]
            valid_nums = image_nums[-100:]
        for file_num in image_nums:
            print("Processing file {subdir} [{file_num}]. "
                  "Full count: {file_count}"
                  .format(subdir=subdir,
                          file_num=file_num,
                          file_count=file_count))
            image_file = os.path.join(full_dir, IMAGE_FILE_FORMAT.format(file_num))
            mat_file = os.path.join(full_dir, MAT_FILE_FORMAT.format(file_num))
            file_count = process_image(image_file, mat_file, out_dir, file_count, remove)
        if subdir == "Train":
            file_count = 0
            out_dir = os.path.join(output_dir, "Valid")
            os.makedirs(out_dir, exist_ok=True)
            for file_num in valid_nums:
                print("Processing file Valid [{file_num}/{max_num}]. "
                      "Full count: {file_count}"
                      .format(subdir=subdir,
                              file_num=file_num,
                              max_num=max_num,
                              file_count=file_count))
                image_file = os.path.join(full_dir, IMAGE_FILE_FORMAT.format(file_num))
                mat_file = os.path.join(full_dir, MAT_FILE_FORMAT.format(file_num))
                file_count = process_image(image_file, mat_file, out_dir, file_count, remove)


def process_image(image_file, mat_file, out_dir, file_count, remove):
    image = cv2.imread(image_file)
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image_grey.shape[0] < WINDOW_SHAPE:
        diffx = WINDOW_SHAPE - image_grey.shape[0]
        image_grey = np.pad(image_grey, ((diffx, 0), (0, 0)), 'constant',
                            constant_values=0)
    if image_grey.shape[1] < WINDOW_SHAPE:
        diffy = WINDOW_SHAPE - image_grey.shape[1]
        image_grey = np.pad(image_grey, ((0, 0), (diffy, 0)), 'constant',
                            constant_values=0)

    marks = get_marks_from_file(mat_file, image_grey.shape)

    sub_images = skimage.util.view_as_windows(image_grey, WINDOW_SHAPE, step=WINDOW_SHAPE)
    sub_marks = skimage.util.view_as_windows(marks, WINDOW_SHAPE, step=WINDOW_SHAPE)
    for x in range(0, sub_images.shape[0]):
        for y in range(0, sub_images.shape[1]):
            sub_mark = sub_marks[x, y]
            sub_image = sub_images[x, y]
            out_file = os.path.join(out_dir, str(file_count))
            label = np.atleast_2d(np.count_nonzero(sub_mark))
            np.savez(out_file, image=sub_image, label=label)
            file_count += 1
    if remove:
        os.remove(image_file)
        os.remove(mat_file)
    return file_count