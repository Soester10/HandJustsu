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

from torchsummary import summary

# custom imports
from utils import train, test, optimizers, dataloader
from models import VisionTransformer


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
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = model
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    best_train_acc = 0
    best_acc = 0
    train_accs = []
    test_accs = []

    optimizer = optimizers.choose_optimizer(optimizer_, net, lr, momentum, weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(epochs):
        print(f"Running Epoch {epoch}...")
        train_acc, train_loss, best_train_acc = train.train(
            epoch, optimizer, net, best_train_acc, criterion, trainloader, device
        )
        test_acc, test_loss, best_acc = test.test(
            epoch, optimizer, net, best_acc, criterion, testloader, device, save_weights
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

    print("\n\nBest accuracy:", best_acc)

    if ret_polt_values:
        return train_accs, test_accs


# define hyperparameters
optimizer_ = "adam"
epochs = 20
lr = 0.1
momentum = 0.9
weight_decay = 5e-4
save_weights = True
ret_polt_values = True
criterion = nn.CrossEntropyLoss()
trainloader, testloader = dataloader.dataloader()
model = VisionTransformer.vit_model1()

# execute main
if __name__ == "__main__":
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
    )
