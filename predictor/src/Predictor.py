#!/usr/bin/env python
# coding: utf-8

# Core python includes
import gc
import sys
import logging
import os
import math

# Argument parsing
import argparse

# Data preparation
import numpy as np

from sample import SampleDataset

from training_data import split_training_dir_into_training_and_test_dir
from model import load_gameboy_net
from trainer import train
from music_generator import generate_a_song

from parameters import BATCH_SIZE, WINDOW_SIZE

# Pytorch setup
import torch

# Print human-interpretable outputs
torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)

# Set device to GPU if it is available, otherwise CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# The user needs to specify the execution mode ([fresh] model, [train] an
# existing model, or [generate] music using a model). The user needs to
# specify a model directory, a directory containing training data, and
# a directly that contains out of training sample test-data to synthesize
# new music from.

parser = argparse.ArgumentParser(
    description="Train a model or generate a song with an existing model"
)

parser.add_argument("--mode", required=True)
parser.add_argument("--model-dir", required=True)
parser.add_argument("--data", required=True)
parser.add_argument("--source-dir", required=False)
parser.add_argument("--output-path", required=False)
parser.add_argument("--override-model", required=False)

args = parser.parse_args()

model = load_gameboy_net

mode = args.mode
model_dir = args.model_dir


train_data = args.data + "/train.pt"
test_data = args.data + "/test.pt"


def load_a_dataset(path):
    return torch.utils.data.DataLoader(
        SampleDataset(path, window_size=WINDOW_SIZE, start_at_sample=False),
        num_workers=1,
        batch_size=BATCH_SIZE,
    )


def train_from(path):
    # Create a standard data loader from our samples
    loader = load_a_dataset(train_data)
    test_loader = load_a_dataset(test_data)

    # Train a model with the data loader
    train(loader, test_loader, model, model_dir, path, device)


def generate_from(model_path, output_path):

    # This loader is used as a seed to the NN and needs to
    # start on a complete sample (start_at_sample=True)
    # because we need to know which byte we are in within
    # the current sample when we generate new samples byte
    # by byte
    # We do not always train exactly on a sample when
    # training, which is why this is a flag.
    out_of_sample_loader = torch.utils.data.DataLoader(
        SampleDataset(
            test_data,
            window_size=WINDOW_SIZE,
            start_at_sample=True,
            entire_sample=True,
        )
    )

    # Generate a song using the out of sample loader
    generate_a_song(out_of_sample_loader, model, model_path, device, output_path)


if mode == "fresh":
    train_from(None)
elif mode == "train":
    train_from("last.checkpoint")
elif mode == "generate":
    assert args.output_path is not None
    path = model_dir + "last.checkpoint"
    if args.override_model is not None:
        path = args.override_model
    print("Loading model at:", path)
    generate_from(path, args.output_path)
elif mode == "split_data":
    split_training_dir_into_training_and_test_dir(args.source_dir, args.data)
