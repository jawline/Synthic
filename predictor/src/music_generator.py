import torch
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical

import numpy as np

from sample import (
    BYTES_PER_ENTRY,
    command_of_bytes,
    command_to_bytes,
    print_feature,
)

from parameters import TRAIN_MARKED_TO_NEXT_SAMPLE, WINDOW_SIZE
from moving_window import MovingWindow


def nearest_multiple(x, base):
    return base * round(x / base)


def prepare_seed(loader, command_generator, device, output_path):

    seed = next(iter(loader))[0]

    # Write the seed values out to a file for debugging
    with open(output_path + "/seed.txt", "w") as f:
        for i in range(0, len(seed), BYTES_PER_ENTRY):
            cmd = command_of_bytes(seed[i : i + BYTES_PER_ENTRY])
            print_feature(cmd, file=f)

    # We configure out dataloader to return the entire sample so we
    # can check for overfitting. Prune that sample down here.
    seed = seed[0 : BYTES_PER_ENTRY * WINDOW_SIZE]

    return MovingWindow(seed, device)


def generate_a_song(loader, load_fn, path, device, output_path):

    # A convenience reference to the CPU
    cpu = torch.device("cpu")

    # Load an instance of the model
    command_generator, _, _, _ = load_fn(path, device)
    command_generator = command_generator.eval()

    # Prepare a seed input from the data loader
    window = prepare_seed(loader, command_generator, device, output_path)

    # Sanity check that we have a valid seed
    print("Sanity checking seed")
    for sample in window.window().reshape((-1, BYTES_PER_ENTRY)):
        command_of_bytes(sample)
    print("Sanity check done")

    with open(output_path + "/output.txt", "w") as f:

        predicted = 0

        for i in range(BYTES_PER_ENTRY * 10000):

            # Depending on how far we are predicting in the future from t
            # we adjust the amount of the output that we consider to be
            # a prediction
            stride = 1

            if TRAIN_MARKED_TO_NEXT_SAMPLE:
                stride = BYTES_PER_ENTRY

            # with torch.cuda.amp.autocast():
            model_input = window.window().unsqueeze(0).detach()
            preds = command_generator.predict(model_input).detach()
            preds = preds[0][-stride:]

            for pred in preds:
                pred = pred.type(torch.float32)
                pred = (
                    RelaxedOneHotCategorical(temperature=1.0, logits=pred)
                    .sample()
                    .argmax()
                )
                window.append(pred)
                predicted += 1

                should_output_sample = predicted % BYTES_PER_ENTRY == 0

                if should_output_sample:
                    try:
                        last_sample = (
                            window.window()[-BYTES_PER_ENTRY:]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.uint8)
                        )
                        print(last_sample)
                        last_sample = command_of_bytes(last_sample)
                        print_feature(last_sample, file=f)
                    except BaseException as err:
                        print("pred was not valid because:", err)
                        raise Exception("predictions stopped looking valid")
