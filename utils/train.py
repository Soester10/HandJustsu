from tqdm import tqdm

## train function for model training with regular batch size
def train(epoch, optimizer, net, best_train_acc, criterion, trainloader, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets, video_path) in tqdm(enumerate(trainloader)):
        #permutating input wrt model input requirement
        inputs = inputs.permute(0, 2, 1, 3, 4)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print("train:", train_loss / (batch_idx + 1), correct / total, end="\r")

    #calculating test accuracy
    if best_train_acc < 100.0 * correct / total:
        best_train_acc = 100.0 * correct / total

    return 100.0 * correct / total, train_loss / (batch_idx + 1), best_train_acc

## train function for model training with regular batch size
def train_mul(
    epoch,
    optimizer,
    net,
    best_train_acc,
    criterion,
    trainloader,
    device,
    batch_multiplier,
):
    """
    
    Batch Multiplier allows users with limited GPU VRAM
    to train with larger batch size.

    The gradients are calculated and updated after batch_size/ optimul_size
    number of runs. This allows for weight updating after
    processing a largerer batch size

    """
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    count = 0

    for batch_idx, (inputs, targets, video_path) in tqdm(enumerate(trainloader)):
        inputs = inputs.permute(0, 2, 1, 3, 4)
        inputs, targets = inputs.to(device), targets.to(device)

        # gradiend descent
        if count == 0:
            optimizer.step()
            optimizer.zero_grad()
            count = batch_multiplier

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        count -= 1

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print("train_mul:", train_loss / (batch_idx + 1), correct / total, end="\r")

    acc = 100.0 * correct / total
    best_train_acc = max(best_train_acc, acc)

    return acc, train_loss / (batch_idx + 1), best_train_acc
