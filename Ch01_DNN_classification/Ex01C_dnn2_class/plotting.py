def plot_data_1d(x, y_gt):
    import matplotlib.pyplot as plt

    plt.scatter(x, y_gt, s=20, c="k")
    plt.axis("equal")
    plt.xlabel("x0", fontsize=24)
    plt.ylabel("y", fontsize=24)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.show()


def plot_pred_1d(x, y_gt, y_p):
    import matplotlib.pyplot as plt

    plt.scatter(x, y_gt, s=20, c="k", label="groundtruth")
    plt.scatter(x, y_p, s=100, c="tab:orange", marker="x", label="predicted")
    plt.legend(fontsize=20)
    plt.axis("equal")
    plt.xlabel("x0", fontsize=24)
    plt.ylabel("y", fontsize=24)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.show()


def plot_data_2d(x, y_gt):
    import matplotlib.pyplot as plt

    plt.scatter(x[:, 0], x[:, 1], c=y_gt, s=50) 
    plt.colorbar()
    plt.axis("equal")
    plt.xlabel("x0", fontsize=24)
    plt.ylabel("x1", fontsize=24)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.show()


def plot_pred_2d(x, y_gt, y_p):
    import matplotlib.pyplot as plt

    plt.scatter(x[:, 0], x[:, 1], c=y_gt, s=50, label="groundtruth")
    plt.scatter(x[:, 0], x[:, 1], c=y_p, s=100, marker="x", label="predicted")
    plt.legend(fontsize=20)
    plt.colorbar()
    plt.axis("equal")
    plt.xlabel("x0", fontsize=24)
    plt.ylabel("x1", fontsize=24)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.show()
