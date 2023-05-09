import torch
import os
from tqdm import tqdm


def test(net, best_acc, criterion, input, device, classes):
    net.eval()

    with torch.no_grad():
        input = input.permute(0, 2, 1, 3, 4)
        input = input.to(device)
        output = net(input)
        _, predicted = output.max(1)

        class_ = classes[predicted]

    return class_
