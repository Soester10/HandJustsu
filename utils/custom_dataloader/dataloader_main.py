from .custom_dataloader import VideoFrameDataset, ImglistToTensor
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
from torch.utils.data import DataLoader


def plot_video(rows, cols, frame_list, plot_width, plot_height, title: str):
    fig = plt.figure(figsize=(plot_width, plot_height))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
        axes_pad=0.3,  # pad between axes in inch.
    )

    for index, (ax, im) in enumerate(zip(grid, frame_list)):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_title(index)
    plt.suptitle(title)
    plt.show()
    fig.savefig("frame_plot.png")


def get_custom_loader(
    batch_size,
    annotations_file: str = "annotations200.txt",
    videos_root: str = "data",
    get_only_test: bool = False,
):
    """

    # Folder format:
    # base_dir = data
    # data/annotations.txt
    # data/<word: hello, books, etc.>/<id>/<frames>


    The frame index range [START_FRAME, END_FRAME] is divided into
    NUM_SEGMENTS even segments. From each segment, a random start-index
    is sampled from which FRAMES_PER_SEGMENT consecutive indices are loaded.
    This results in NUM_SEGMENTS*FRAMES_PER_SEGMENT chosen indices, whose
    frames are loaded as PIL images and put into a list and returned when
    calling dataset[i].

    If you do not want to use sparse temporal sampling, and instead
    want to just load N consecutive frames starting from a random
    start index, this is easy. Simply set NUM_SEGMENTS=1 and
    FRAMES_PER_SEGMENT=N. Each time a sample is loaded, N
    frames will be loaded from a new random start index.

    As of torchvision 0.8.0, torchvision transforms support batches of images
    of size (BATCH x CHANNELS x HEIGHT x WIDTH) and apply deterministic or random
    transformations on the batch identically on all images of the batch. Any torchvision
    transform for image augmentation can thus also be used  for video augmentation.

    In case a different types of transforms are applied, you can pass them as a list
    to the 'transforms' parameter


    """

    annotation_file = f"{videos_root}/{annotations_file}"

    dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=annotation_file,
        num_segments=40,
        frames_per_segment=1,
        imagefile_template="img_{:03d}.jpg",
        transform=[ImglistToTensor()]
        if get_only_test
        else [
            ImglistToTensor(),
            # transforms.RandomGrayscale(p=0.35),
            # transforms.RandomInvert(p=0.5),
            # transforms.Compose([
            #     ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
            #     transforms.Resize(299),  # image batch, resize smaller edge to 299
            #     transforms.CenterCrop(299),  # image batch, center crop to square 299x299
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # ])
        ],
        test_mode=True,
    )

    if get_only_test:
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1
        )
        return data_loader

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )

    return train_data_loader, test_data_loader
