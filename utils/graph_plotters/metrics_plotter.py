import matplotlib.pyplot as plt


## Plot Losses
def plot_loss(train_losses: list, test_losses: list, epochs: int):
    # train plot
    plt.plot(range(epochs), train_losses, color="tab:blue")
    # test plot
    plt.plot(range(epochs), test_losses, color="tab:orange")

    plt.title("Train and Test Losses vs Epochs")
    plt.legend(["Train Loss", "Test Loss"])

    # display the plot
    plt.show()


## Plot Accuracies
def plot_acc(train_accs: list, test_accs: list, epochs: int):
    # train plot
    plt.plot(range(epochs), train_accs, color="tab:blue")
    # test plot
    plt.plot(range(epochs), test_accs, color="tab:orange")

    plt.title("Train and Test Accuracy vs Epochs")
    plt.legend(["Train Accuracy", "Test Accuracy"])

    # display the plot
    plt.show()
