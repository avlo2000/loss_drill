import matplotlib.pyplot as plt


def plot_loss_drill(weights, losses):
    plt.title("Loss drill before train and after")
    plt.xlabel("ModelA to ModelB lerp weight")
    plt.ylabel("Total cross entropy loss")
    plt.plot(weights, losses, label=f"Loss drill num")
    plt.legend()
    plt.show()
