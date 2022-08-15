from datetime import datetime

import torch
from torch.cuda.amp import autocast
from torch import nn

from sample import MAX_WINDOW_SIZE, BYTES_PER_ENTRY


class EarlyExit:
    def __init__(self, tolerence):
        self.tolerence = tolerence
        self.bad_rounds = 0
        self.last_losses = torch.as_tensor([])

    def mean_loss(self):
        if self.last_losses.size(dim=0) == 0:
            return None

        return self.last_losses.mean()

    def append_loss(self, loss):
        self.last_losses = torch.cat((self.last_losses, torch.as_tensor([loss])))
        self.last_losses = self.last_losses[-8:]

    def update(self, loss):
        mean_loss = self.mean_loss()
        print("Mean validation loss: ", mean_loss)

        if mean_loss is None:
            return

        if loss > mean_loss:
            self.bad_rounds += 1
        else:
            self.bad_rounds = 0
            self.append_loss(loss)

        if self.bad_rounds == self.tolerence:
            raise Exception("early exit because validation loss stopped going down")


ROUND_SZ = 1000
VALIDATION_SZ = 500


def train(data_loader, validation_loader, load_fn, model_dir, load_path, device):

    early_exit = EarlyExit(4)

    cpu = torch.device("cpu")

    print("Train called with: ", model_dir, load_path)

    # Send none to load_fn if load_path is None otherwise append the model dir to it
    path = None

    if load_path != None:
        path = model_dir + load_path

    command_generator, optimizer, scheduler = load_fn(path, device)
    criterion = nn.CrossEntropyLoss()
    running_loss = torch.zeros(1, device=device)

    print("Next data loader")
    data_loader = iter(data_loader)
    validation_loader = iter(validation_loader)

    def step(ldr, sz, backprop):

        print("Starting batch")
        running_loss.zero_()

        for i in range(sz):

            if i % (sz / 4) == 0:
                print("Batch completion:", (float(i) / float(sz)) * 100.0, "%")

            seq = next(ldr).to(device)
            inputs = seq[:, :-BYTES_PER_ENTRY]
            labels = seq[:, BYTES_PER_ENTRY:]

            optimizer.zero_grad()
            logits = command_generator(inputs)

            with torch.cuda.amp.autocast():
                loss = criterion(logits, labels)

            if backprop:
                loss.backward()
                optimizer.step()
            running_loss.add_(loss.detach())

            seq = seq.detach().to(cpu)

        result = running_loss / sz
        return result

    def save(name):
        print("Saving model to path: " + name)
        torch.save(command_generator.state_dict(), "./" + name + ".model")
        torch.save(optimizer.state_dict(), "./" + name + ".optimizer")

    epoch = 1

    while True:

        print("Pre-step LR:", optimizer.param_groups[0]["lr"])

        # Do a ROUND_SZ of training and backprop
        loss = step(data_loader, ROUND_SZ, True)

        # Feed the current epoch and loss (1-indexed not 0-indexed) into our scheduler function to adjust the LR
        scheduler.step(loss)

        # Do a round 10 of validation with no backprop
        validation_loss = step(validation_loader, VALIDATION_SZ, False)

        early_exit.update(validation_loss.item())

        print("Loss:", loss.item())
        print("Validation loss:", validation_loss.item())
        print("LR:", optimizer.param_groups[0]["lr"])

        print("Saving checkpoint")

        # Timestamp every 10th epoch to test fits later
        if epoch % 3 == 0:
            save(model_dir + "/" + str(int(datetime.now().timestamp())))

        save(model_dir + "/last.checkpoint")

        print("Saved checkpoint")

        epoch += 1

    return command_generator.eval()
