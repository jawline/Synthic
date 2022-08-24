import sys
import numpy as np
import os
import time
import random
import torch
import shutil


# These commands enumerate the different kind of instruction we can send to each channel.
# Note: not every command is legal for every channel, invalid commands will be ignored.
CMD_VOLENVPER = 1
CMD_DUTYLL = 2
CMD_MSB = 3
CMD_LSB = 4
CMD_COUNT = 5

# These offsets mark the type of data index in sample form (not model form)
TIME_OFFSET = 0
CH_OFFSET = 1
CMD_OFFSET = 2
PARAM1_OFFSET = 3
PARAM2_OFFSET = 4
PARAM3_OFFSET = 5
SIZE_OF_INPUT_FIELDS = 6

# The maximum number of samples we will send to the model in a single iteration
MAX_WINDOW_SIZE = 1024

# The Gameboy cycles this many times per second. This is the
# measurement of time we use in our TIME_OFFSET values
M_CYCLES_PER_SECOND = 4194304.0

# This is the number of bytes in every individual sample given
# to the model after it is converted to bytes
BYTES_PER_ENTRY = 7


def fresh_input(command, channel, time):
    newd = np.zeros(shape=SIZE_OF_INPUT_FIELDS, dtype=int)
    newd[TIME_OFFSET] = time
    newd[CH_OFFSET] = channel
    newd[CMD_OFFSET] = command
    return newd


def parse_bool(v):
    if v == "true":
        return 1
    elif v == "false":
        return 0
    else:
        return int(v)


def command_of_parts(command, channel, parts, time):
    inp = fresh_input(command, channel, time)

    if command == CMD_DUTYLL:
        inp[PARAM1_OFFSET] = int(parts[3])
        inp[PARAM2_OFFSET] = int(parts[4])
    elif command == CMD_VOLENVPER:
        inp[PARAM1_OFFSET] = int(parts[3])
        inp[PARAM2_OFFSET] = parse_bool(parts[4])
        inp[PARAM3_OFFSET] = int(parts[5])
    elif command == CMD_LSB:
        inp[PARAM1_OFFSET] = int(parts[3])
        inp[PARAM2_OFFSET] = 0
        inp[PARAM3_OFFSET] = 0
    elif command == CMD_MSB:
        inp[PARAM1_OFFSET] = int(parts[3])
        inp[PARAM2_OFFSET] = parse_bool(parts[4])
        inp[PARAM3_OFFSET] = parse_bool(parts[5])
    else:
        raise Exception("this should not happen")
    return inp


def int32_as_bytes(ival):
    return np.frombuffer(ival.item().to_bytes(4, byteorder="big"), dtype=np.uint8)


def int32_of_bytes(np):
    return int.from_bytes(np, byteorder="big")


def int8_as_bytes(ival):
    return np.frombuffer(ival.item().to_bytes(1, byteorder="big"), dtype=np.uint8)


def int8_of_bytes(np):
    return int.from_bytes(np, byteorder="big")


def merge_params(data):
    command = data[CMD_OFFSET]
    if command == CMD_DUTYLL:
        return (data[PARAM1_OFFSET] << 6) | data[PARAM2_OFFSET]
    elif command == CMD_VOLENVPER:
        return (
            (data[PARAM1_OFFSET] << 4)
            | (data[PARAM2_OFFSET] << 3)
            | data[PARAM3_OFFSET]
        )
    elif command == CMD_LSB:
        return data[PARAM1_OFFSET]
    elif command == CMD_MSB:
        return (
            data[PARAM1_OFFSET]
            | (data[PARAM2_OFFSET] << 6)
            | (data[PARAM3_OFFSET] << 7)
        )
    else:
        raise Exception("this should not happen")


def unmerge_params(command, data, v):
    if command == CMD_DUTYLL:
        data[PARAM1_OFFSET] = v >> 6
        data[PARAM2_OFFSET] = v & 0b0011_1111
    elif command == CMD_VOLENVPER:
        data[PARAM1_OFFSET] = v >> 4
        data[PARAM2_OFFSET] = (v & 0b0000_1000) >> 3
        data[PARAM3_OFFSET] = v & 0b0000_0111
    elif command == CMD_LSB:
        data[PARAM1_OFFSET] = v
    elif command == CMD_MSB:
        data[PARAM1_OFFSET] = v & 0b0011_1111
        data[PARAM2_OFFSET] = (v & 0b0100_0000) >> 6
        data[PARAM3_OFFSET] = (v & 0b1000_0000) >> 7
    else:
        raise Exception("this should not happen")


def command_to_bytes(command):
    new_arr = np.concatenate(
        [
            int32_as_bytes(command[TIME_OFFSET]),
            int8_as_bytes(command[CH_OFFSET]),
            int8_as_bytes(command[CMD_OFFSET]),
            int8_as_bytes(merge_params(command)),
        ]
    ).flatten()
    return new_arr


def command_of_bytes(byte_arr):
    d = fresh_input(0, 0, 0)
    d[TIME_OFFSET] = int32_of_bytes(byte_arr[0:4])
    d[CH_OFFSET] = int8_of_bytes(byte_arr[4:5])
    if d[CH_OFFSET] != 1 and d[CH_OFFSET] != 2:
        raise Exception("bad channel prediction " + str(d[CH_OFFSET]))
    d[CMD_OFFSET] = int8_of_bytes(byte_arr[5:6])
    unmerge_params(d[CMD_OFFSET], d, byte_arr[6])
    return d


def print_feature(data, file=sys.stdout):
    command = data[CMD_OFFSET]
    if command == CMD_DUTYLL:
        print(
            f"CH {data[CH_OFFSET]} DUTYLL {data[PARAM1_OFFSET]} {data[PARAM2_OFFSET]} AT {data[TIME_OFFSET]}",
            file=file,
            flush=True,
        )
    elif command == CMD_VOLENVPER:
        print(
            f"CH {data[CH_OFFSET]} VOLENVPER {data[PARAM1_OFFSET]} {data[PARAM2_OFFSET]} {data[PARAM3_OFFSET]} AT {data[TIME_OFFSET]}",
            file=file,
            flush=True,
        )
    elif command == CMD_LSB:
        print(
            f"CH {data[CH_OFFSET]} FREQLSB {data[PARAM1_OFFSET]} AT {data[TIME_OFFSET]}",
            file=file,
            flush=True,
        )
    elif command == CMD_MSB:
        print(
            f"CH {data[CH_OFFSET]} FREQMSB {data[PARAM1_OFFSET]} {data[PARAM2_OFFSET]} {data[PARAM3_OFFSET]} AT {data[TIME_OFFSET]}",
            file=file,
            flush=True,
        )
    else:
        print(f"Bad prediction", file=file, flush=True)


def load_training_data(src):
    data = []
    file = open(src, "r")
    for line in file:
        parts = line.split()
        if len(parts) > 0 and parts[0] == "CH":
            # print(parts)
            channel = int(parts[1])
            command = parts[2]
            time = int(parts[-1])
            if command == "DUTYLL":
                new_item = command_of_parts(CMD_DUTYLL, channel, parts, time)
            elif command == "VOLENVPER":
                new_item = command_of_parts(CMD_VOLENVPER, channel, parts, time)
            elif command == "FREQLSB":
                new_item = command_of_parts(CMD_LSB, channel, parts, time)
            elif command == "FREQMSB":
                new_item = command_of_parts(CMD_MSB, channel, parts, time)
            else:
                # Otherwise unknown data we ignore. We currently don't handle some commands (e.g, sweep) so erroring here is spurious
                print("Unknown", command)
            data.append(new_item)
    return data


def load_raw_data(src, window_size):

    sample_data = load_training_data(src)
    sample_data = np.array([command_to_bytes(x) for x in sample_data]).flatten()

    # If the sample is less than the window size then ignore it
    # TODO: Left pad again?
    if len(sample_data) < (window_size * BYTES_PER_ENTRY) * 2:
        raise Exception("Bad file")

    return torch.Tensor(sample_data).long()


class SampleDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, window_size, start_at_sample=False, max_files=None):
        super(SampleDataset).__init__()
        files = self.training_files(path)

        print("Training files: ")
        for filename in files:
            print(filename)

        # Select a random set of files if max_files is set
        if max_files is not None:
            random.shuffle(files)
            files = files[0:max_files]

        # Load the files and convert them to the model encoding
        self.window_size = window_size
        self.file_datas = self.load_sample_files(files)

        print("Created the loader")

    def training_files(self, dirp):
        return [
            os.path.join(root, fname)
            for (root, dir_names, file_names) in os.walk(dirp, followlinks=True)
            for fname in file_names
        ]

    def load_sample_files(self, files):
        file_datas = []

        for file in files:
            print("Loading ", file)
            try:
                file_datas.append((file, load_raw_data(file, self.window_size)))
            except Exception as e:
                print("Caught and ignoring file exception: ", e)

        print("Done loading streams")

        return file_datas

    def random_start_offset(self, sample_data):

        # Randomly start from a offset in window size to change the interleaving of data
        # at each pass of the file
        start_idx = np.random.randint(0, self.window_size)

        # print("Skipping: ", start_idx)
        # print("Start offset: ", BYTES_PER_ENTRY * start_idx)

        return sample_data[start_idx:]

    def __iter__(self):
        print("Iterating over dataset")

        idx = 0

        random.shuffle(self.file_datas)
        print("Shuffled dataset for epoch")

        for (name, data) in self.file_datas:

            idx += 1
            data = data.reshape((-1, BYTES_PER_ENTRY))
            data = self.random_start_offset(data)
            total_samples = data.shape[0]
            rounds = int(total_samples / self.window_size)

            # print("Iterating over: ", name)
            # print("Total samples: ", total_samples)
            # print("Rounds: ", rounds)

            if idx % (total_samples / 10) == 0:
                print("Epoch %", (float(idx) / float(total_samples)) * 100)

            for round_idx in range(rounds):
                # print("Round: ", round_idx)

                # Select one additional sample for the output we are predicting.
                start_index_in_samples = round_idx
                end_index_in_samples = round_idx + self.window_size + 1

                next_data = data[start_index_in_samples:end_index_in_samples].reshape(
                    (-1)
                )

                yield next_data


def copy_file(src_file, dst_dir):
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
        copy_file(f, out_dir_training)

    for f in test_files:
        copy_file(f, out_dir_testing)
