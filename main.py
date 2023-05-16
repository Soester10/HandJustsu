# Lib imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys

from torchsummary import summary

# custom imports
from utils import train, test, optimizers, dataloader, custom_test
from utils.custom_dataloader import custom_dataloader, dataloader_main
from models import VisionTransformer

##to resolve 'ssl certificate verify failed' issue
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

##to resolve " No module named 'custom_dataloader' "
sys.path.append("utils/custom_dataloader")


def main(
    criterion,
    optimizer_,
    epochs,
    lr,
    momentum,
    weight_decay,
    save_weights,
    ret_polt_values,
    trainloader,
    testloader,
    model,
    train_mul,
    batch_size,
    optimal_batch_size=8,
    load_from_ckpt=False,
    labeled_test=True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = model
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    start_epoch = 0
    best_train_acc = 0
    best_acc = 0
    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []

    if load_from_ckpt and os.path.isfile("checkpoint/ckpt.pth"):
        checkpoint = torch.load("checkpoint/ckpt.pth")
        net.load_state_dict(checkpoint["net"])
        best_acc = checkpoint["best_acc"]
        start_epoch = checkpoint["epoch"] + 1
        train_accs = checkpoint["train_accs"]
        test_accs = checkpoint["test_accs"]
        train_losses = checkpoint["train_losses"]
        test_losses = checkpoint["test_losses"]

    optimizer = optimizers.choose_optimizer(optimizer_, net, lr, momentum, weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    if train_mul:
        batch_multiplier = batch_size // optimal_batch_size

    for epoch in range(start_epoch, start_epoch + epochs):
        print(f"Running Epoch {epoch}...")

        if train_mul:
            train_acc, train_loss, best_train_acc = train.train_mul(
                epoch,
                optimizer,
                net,
                best_train_acc,
                criterion,
                trainloader,
                device,
                batch_multiplier,
            )
        else:
            train_acc, train_loss, best_train_acc = train.train(
                epoch, optimizer, net, best_train_acc, criterion, trainloader, device
            )

        test_acc, test_loss, best_acc = test.test(
            epoch, optimizer, net, best_acc, criterion, testloader, device
        )

        scheduler.step()

        print(
            "Train==>",
            "Loss:",
            format(train_loss, ".03f"),
            "Acc:",
            format(train_acc, ".03f"),
        )
        print(
            "Test==>",
            "Loss:",
            format(test_loss, ".03f"),
            "Acc:",
            format(test_acc, ".03f"),
        )

        train_accs.append(train_acc)
        test_accs.append(test_acc)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if best_acc <= test_acc and save_weights:
            print("Saving Weights...")
            best_acc = max(best_acc, test_acc)
            state = {
                "net": net.state_dict(),
                "best_acc": best_acc,
                "epoch": epoch,
                "train_accs": train_accs,
                "test_accs": test_accs,
                "train_losses": train_losses,
                "test_losses": test_losses,
            }
            if not os.path.isdir("checkpoint"):
                os.mkdir("checkpoint")
            torch.save(state, "checkpoint/ckpt.pth")

        print("Saving Final Weights...")
        state = {
            "net": net.state_dict(),
            "best_acc": best_acc,
            "epoch": epoch,
            "train_accs": train_accs,
            "test_accs": test_accs,
            "train_losses": train_losses,
            "test_losses": test_losses,
        }
        torch.save(state, "checkpoint/ckpt_final.pth")

    print("\n\nBest Test Accuracy:", best_acc)

    if ret_polt_values:
        return train_accs, test_accs


def custom_tester(model, path_to_videos, labeled_test):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = model
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    try:
        checkpoint = torch.load("checkpoint/ckpt.pth")
    except:
        print("Checkpoint not Found!")
        return
    net.load_state_dict(checkpoint["net"])

    predicted_classes, correct = custom_test.test(
        net, path_to_videos, device, known=labeled_test
    )

    print(predicted_classes)
    if labeled_test:
        print(f"Accuracy => {correct}%")

    if not os.path.isdir("output"):
        os.mkdir("output")

    with open("output/output.txt", "w") as f:
        for path_ in predicted_classes:
            to_write: str = (
                str(path_)
                + "=> predicted: "
                + str(predicted_classes[path_][0])
                + " | real: "
                + str(predicted_classes[path_][1])
                + "\n"
            )
            f.write(to_write)


## CL Arguments
parser = argparse.ArgumentParser(description="HandJutsu")
# testing
parser.add_argument("--test", action="store_true", help="to test the model")
parser.add_argument(
    "--test_data_path",
    default="./test_data/sample_test_data",
    type=str,
    help="test data path",
)
parser.add_argument(
    "--unlabeled_test",
    action="store_true",
    help="to know the dataset is labeled or unlabeled",
)
# training
parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
parser.add_argument(
    "--scratch", action="store_true", help="trains from scratch without ckpt"
)
args = parser.parse_args()


# define hyperparameters
batch_size: int = 32
optimal_batch_size: int = 8
train_mul: bool = True
optimizer_: str = "adadelta"
epochs: int = args.epochs
lr: float = 0.1
momentum: float = 0.9
weight_decay: float = 5e-4
save_weights: bool = True
ret_polt_values: bool = True
criterion = nn.CrossEntropyLoss()

model = VisionTransformer.vivit_model1(
    image_size=200,
    frames=40,
    num_classes=2000,
)

## WLASL data loader
trainloader, testloader = dataloader_main.get_custom_loader(
    optimal_batch_size if train_mul else batch_size,
    annotations_file="annotations200.txt",
)
print(len(trainloader), len(testloader))

load_from_ckpt: bool = not (args.scratch)


## Custom Test
custom_test_: bool = args.test
path_to_videos: str = args.test_data_path  # path to video/s
labeled_test: bool = not (args.unlabeled_test)

# execute main
if __name__ == "__main__":
    if custom_test_:
        custom_tester(model, path_to_videos, labeled_test)
        sys.exit(0)
    main(
        criterion,
        optimizer_,
        epochs,
        lr,
        momentum,
        weight_decay,
        save_weights,
        ret_polt_values,
        trainloader,
        testloader,
        model,
        train_mul,
        batch_size,
        optimal_batch_size,
        load_from_ckpt,
        labeled_test,
    )
