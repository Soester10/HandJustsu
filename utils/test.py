import torch
import os
from tqdm import tqdm


def test(epoch, optimizer, net, best_acc, criterion, testloader, device, save_weights):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.0 * correct / total
    if acc > best_acc:
        if save_weights:
            print("Saving..")
            state = {
                "net": net.state_dict(),
                "acc": acc,
                "epoch": epoch,
            }
            if not os.path.isdir("checkpoint"):
                os.mkdir("checkpoint")
            torch.save(state, "./checkpoint/ckpt.pth")

        best_acc = acc

    return 100.0 * correct / total, test_loss / (batch_idx + 1), best_acc
