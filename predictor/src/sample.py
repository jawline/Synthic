import sys
import numpy as np
import os
import time
import random
import torch
import shutil
from termcolor import colored

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
MAX_WINDOW_SIZE = 256

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
    if len(sample_data) < (window_size * BYTES_PER_ENTRY):
        raise Exception("Bad file")

    return torch.Tensor(sample_data).long()


class SampleDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        path,
        window_size,
        start_at_sample=False,
        max_files=None,
        entire_sample=False,
    ):
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
        self.start_at_sample = start_at_sample
        self.entire_sample = entire_sample
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
                new_file_data = load_raw_data(file, self.window_size)
                file_datas.append((file, new_file_data))
            except Exception as e:
                print("Caught and ignoring file exception: ", e)

        print("Done loading streams")

        return file_datas

    def random_start_offset(self, sample_data):
        def lround(x):
            return int(7 * round(float(x) / 7))

        bytes_per_window_size = BYTES_PER_ENTRY * self.window_size
        max_start_idx = sample_data.shape[0] - bytes_per_window_size

        start_idx = 0

        if max_start_idx != 0:
            start_idx = np.random.randint(0, max_start_idx)

        if self.start_at_sample:
            start_idx = lround(start_idx)

        end_idx = start_idx + bytes_per_window_size

        if self.entire_sample:
            return sample_data[start_idx:]
        else:
            return sample_data[start_idx : start_idx + bytes_per_window_size]

    def __iter__(self):

        epoch_data = []

        for (name, data) in self.file_datas:
            for _i in range(64):
                next_step_data = self.random_start_offset(data)
                epoch_data.append(next_step_data)

        random.shuffle(epoch_data)

        print("Iterating over", len(epoch_data), "random samples from the dataset")

        count = 0
        for data in epoch_data:
            count += 1
            if count % int(len(epoch_data) / 10) == 0:
                print(
                    colored(
                        "Epoch: " + str((count / len(epoch_data)) * 100) + "%",
                        "green",
                        attrs=[],
                    )
                )
            yield data
