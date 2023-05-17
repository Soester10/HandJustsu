import torch
import os
from tqdm import tqdm


## test function for model performance validation
def test(epoch, optimizer, net, best_acc, criterion, testloader, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets, video_path) in tqdm(enumerate(testloader)):
            # permutating input wrt model input requirement
            inputs = inputs.permute(0, 2, 1, 3, 4)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print("test:", test_loss / (batch_idx + 1), correct / total, end="\r")

    # calculating test accuracy
    acc = 100.0 * correct / total

    return acc, test_loss / (batch_idx + 1), best_acc
