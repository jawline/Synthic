import torch
from torch.distributions.categorical import Categorical

import numpy as np

from sample import BYTES_PER_ENTRY, command_of_bytes, command_to_bytes, print_feature

# This block allocates some linear memory we
# feed the new predictions into and use as a
# rolling window for the NN so we don't need
# to memmove as often
class MovingWindow():

    def __init__(self, seed, device):

        # Pre-allocate 16x the seed
        self.seq = torch.cat((seed.long(), torch.zeros(len(seed) * 16).long())).to(device)

        self.start = 0
        self.len = len(seed)

    def end(self):
        return self.start + self.len

    def append(self, item):

        # when we run out of free slots we move the array by using torch.roll
        # so that the data we care about is from 0:len again.
        if self.end() == len(self.seq):
            # Roll of a 1d Tensor => arr[i] = arr[(i + shift) % len(arr)], so the most recent element 
            torch.roll(self.seq, self.len)
            self.start = 0
        else:
            self.seq[self.end()] = item
            self.start += 1

    def window(self):
        # Slice the current window
        return self.seq[self.start:self.end()]

def prepare_seed(loader, command_generator, device):

  # Cut the seed to the receptive window of our model so that it executes faster
  seed = next(iter(loader))[0][:command_generator.receptive_field()]

  # Write the seed values out to a file for debugging
  with open('seed.txt', 'w') as f:
      for i in range(0, len(seed), BYTES_PER_ENTRY):
          print("Seed value :", i, seed.shape)
          cmd = command_of_bytes(seed[i:i+BYTES_PER_ENTRY])
          print_feature(cmd, file=f)

  return MovingWindow(seed, device)

def generate_a_song(loader, load_fn, path, device):

  # A convienience reference to the CPU
  cpu = torch.device('cpu')

  # Load an instance of the model
  command_generator, _, _ = load_fn(path, device)
  command_generator = command_generator.eval()

  # Prepare a seed input from the data loader
  window = prepare_seed(loader, command_generator, device)

  with open('output.txt', 'w') as f:
      for i in range(BYTES_PER_ENTRY * 10000):
          seq = window.window().unsqueeze(0)
          pred = command_generator(seq).detach().to(cpu).permute(0,2,1).squeeze(0)[-1]
          pred = Categorical(logits=pred).sample()
          window.append(pred)

          if (i + 1) % BYTES_PER_ENTRY == 0:
              try:
                  last_sample = window.window()[-BYTES_PER_ENTRY:].detach().cpu().numpy().astype(np.uint8)
                  last_sample = command_of_bytes(last_sample)
                  print_feature(last_sample, file=f)
              except BaseException as err:
                  print("pred was not valid because:", err)

      del pred