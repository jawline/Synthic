import shutil
import os
import random


def copy_file(src_file, dst_dir):
    print("Make", dst_dir + "/" + os.path.dirname(src_file))
    os.makedirs(dst_dir + "/" + os.path.dirname(src_file), exist_ok=True)
    shutil.copyfile(src_file, dst_dir + "/" + src_file)


def split_training_dir_into_training_and_test_dir(
    in_dir, out_dir_training, out_dir_testing
):
    files = [
        os.path.join(root, fname)
        for (root, dir_names, file_names) in os.walk(in_dir, followlinks=True)
        for fname in file_names
    ]
    print("80/20 split of: " + str(len(files)) + " files")

    random.shuffle(files)

    num_files_train = int(len(files) * 0.8)

    train_files = files[0:num_files_train]
    test_files = files[num_files_train:]

    for f in train_files:
        print("Copy " + f)
        copy_file(f, out_dir_training)

    for f in test_files:
        copy_file(f, out_dir_testing)
