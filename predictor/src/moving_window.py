import torch

"""
This block allocates some linear memory we feed new predictions into and use as a
rolling window for the NN so we don't need to move memory as often.
"""


class MovingWindow:
    def __init__(self, seed, device):

        # Pre-allocate 16x the seed
        self.seq = torch.cat((seed.long(), torch.zeros(len(seed) * 16).long())).to(
            device
        )

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
        return self.seq[self.start : self.end()]
