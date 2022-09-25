import torch


class EarlyExit:
    def __init__(self, lookback_window, tolerence):
        self.tolerence = tolerence
        self.lookback_window = lookback_window
        self.bad_rounds = 0
        self.last_losses = torch.as_tensor([])

    def mean_loss(self):
        if self.last_losses.size(dim=0) == 0:
            return None

        return self.last_losses.mean()

    def append_loss(self, loss):
        self.last_losses = torch.cat((self.last_losses, torch.as_tensor([loss])))
        self.last_losses = self.last_losses[-self.lookback_window :]

    def update(self, loss):
        mean_loss = self.mean_loss()
        print("Mean validation loss: ", mean_loss)

        if mean_loss is None:
            self.append_loss(loss)
            return

        if loss > mean_loss:
            self.bad_rounds += 1
        else:
            self.bad_rounds = 0
            self.append_loss(loss)

        if self.bad_rounds == self.tolerence:
            raise Exception("early exit because validation loss stopped going down")
