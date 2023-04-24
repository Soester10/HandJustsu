import torch.optim as optim


def choose_optimizer(optimizer_, net, lr=0.1, momentum=0.9, weight_decay=5e-4):
    optimizers = {
        "sgd": optim.SGD,
        "adam": optim.Adam,
        "sgd-nesterov": optim.SGD,
        "adagrad": optim.Adagrad,
        "adadelta": optim.Adadelta,
    }

    if optimizer_ == "sgd":
        optimizer = optimizers[optimizer_](
            net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )

    elif optimizer_ == "sgd-nesterov":
        optimizer = optimizers[optimizer_](
            net.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )

    else:
        optimizer = optimizers[optimizer_](
            net.parameters(), lr=lr, weight_decay=weight_decay
        )

    return optimizer
