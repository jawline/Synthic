import shutil
import os
import random
import torch
import numpy as np
from tqdm import tqdm
from sample import load_raw_data


def copy_file(src_file, dst_dir):
    print("Make", dst_dir + "/" + os.path.dirname(src_file))
    os.makedirs(dst_dir + "/" + os.path.dirname(src_file), exist_ok=True)
    shutil.copyfile(src_file, dst_dir + "/" + src_file)


def list_all_files(dirp):
    return [
        os.path.join(root, fname)
        for (root, dir_names, file_names) in os.walk(dirp, followlinks=True)
        for fname in file_names
    ]


def load_sample_files(files):

    file_datas = []

    for i in tqdm(range(len(files))):
        file = files[i]
        try:
            new_file_data = load_raw_data(file, 2000)
            file_datas.append((file, new_file_data))
        except Exception as e:
            print("Caught and ignoring file exception: ", e)

    print("Done loading streams")

    return file_datas


def split_training_dir_into_training_and_test_dir(in_dir, out_dir):

    files = list_all_files(in_dir)

    print("80/20 split of: " + str(len(files)) + " files")

    random.shuffle(files)

    num_files_train = int(len(files) * 0.8)

    train_files = files[0:num_files_train]
    test_files = files[num_files_train:]

    print("Loading training data")

    train_data = load_sample_files(train_files)

    print("Loading testing data")

    test_data = load_sample_files(test_files)

    print("Serializing")

    torch.save(train_data, out_dir + "/train.pt")
    torch.save(test_data, out_dir + "/test.pt")
