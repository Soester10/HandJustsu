def train(epoch, optimizer, net, best_train_acc, criterion, trainloader, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
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

    if best_train_acc < 100.0 * correct / total:
        best_train_acc = 100.0 * correct / total

    return 100.0 * correct / total, train_loss / (batch_idx + 1), best_train_acc