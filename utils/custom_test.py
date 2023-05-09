import torch
import os
from tqdm import tqdm
import json
from . import process_videos_to_format


def test(net, input_: str, device):
    net.eval()

    with torch.no_grad():
        input_ = input_.permute(0, 2, 1, 3, 4)
        input_ = input_.to(device)
        output = net(input_)
        _, predicted = output.max(1)

        classes = open("./labels/label_to_word.json")
        classes = json.load(classes)

        class_ = classes[predicted]

    return class_
