import torch
import torch.nn as nn

from . import resnet


def skyline_model_provider():
    return resnet.resnet50().cuda()


def skyline_input_provider(batch_size=16):
    return (
        torch.randn((batch_size, 3, 50, 50)).cuda(),
        torch.randint(low=0, high=1000, size=(batch_size,)).cuda(),
    )


def skyline_iteration_provider(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    def iteration(*inputs):
        optimizer.zero_grad()
        out = model(*inputs)
        out.backward()
        optimizer.step()
    return iteration
