import torch
import os
from tqdm import tqdm
import json
from .process_videos_to_format import convert_known_videos_to_frames
from .custom_dataloader.dataloader_main import get_custom_loader


def test(net, path_to_videos: str, device, known=True):
    net.eval()

    predicted_classes = {}
    classes = open("utils/labels/label_to_word.json")
    classes = json.load(classes)

    with torch.no_grad():
        if known:
            convert_known_videos_to_frames(path_to_videos)
            custom_testloader = get_custom_loader(
                batch_size=1,
                annotations_file=f"{path_to_videos}/annotations.txt",
                videos_root=path_to_videos,
                get_only_test=True,
            )

        ##TODO
        # else:
        # convert_unkonwn_videos_to_frames(path_to_videos)

        for batch_idx, (input_, targets) in tqdm(enumerate(custom_testloader)):
            input_ = input_.permute(0, 2, 1, 3, 4)
            input_ = input_.to(device)
            output = net(input_)
            _, predicted = output.max(1)

            class_ = classes[predicted]

            ##TODO: change batch_idx to record.path
            predicted_classes[batch_idx] = class_

    return predicted_classes
