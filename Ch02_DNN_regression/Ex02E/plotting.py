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
    plt.colorbar()
    plt.scatter(x[:, 0], x[:, 1], c=y_p, s=100, marker="x", label="predicted")
    plt.legend(fontsize=20)
    plt.axis("equal")
    plt.xlabel("x0", fontsize=24)
    plt.ylabel("x1", fontsize=24)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.show()



def plot_pred_vs_gt(y_gt, y_p):
    import matplotlib.pyplot as plt

    plt.plot([-1, 1], [-1, 1], "k:")
    plt.scatter(y_gt, y_p, s=10) 
    plt.axis("square")
    plt.xlabel("y groundtruth", fontsize=24)
    plt.ylabel("y predicted", fontsize=24)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.show()

def plot_mse(mse, smooth=11):
    import matplotlib.pyplot as plt
    from numpy import convolve, full

    mse_smooth = convolve(mse, full((smooth,), 1 / smooth), mode="valid")

    fig, ax = plt.subplots(1, 2)
    
    fig.set_size_inches(10, 5)
    
    ax[0].plot(mse, c="tab:orange")
    ax[0].plot(range(smooth // 2, len(mse) - smooth // 2), mse_smooth, c="k")
    ax[0].set_xlabel("epoch", fontsize=24)
    ax[0].set_ylabel("MSE", fontsize=24)
    ax[0].tick_params(axis="both", which="major", labelsize=16)

    ax[1].loglog(mse, c="tab:orange")
    ax[1].loglog(range(smooth // 2, len(mse) - smooth // 2), mse_smooth, c="k")
    ax[1].set_xlabel("epoch", fontsize=24)
    ax[1].set_ylabel("MSE", fontsize=24)
    ax[1].tick_params(axis="both", which="major", labelsize=16)
    
    plt.tight_layout()
    plt.show()

def plot_mse_train_vs_val(mse_t, mse_v, smooth=11):
    import matplotlib.pyplot as plt
    from numpy import convolve, full

    mse_t_s = convolve(mse_t, full((smooth,), 1 / smooth), mode="valid")
    mse_v_s = convolve(mse_v, full((smooth,), 1 / smooth), mode="valid")

    fig, ax = plt.subplots(1, 2)
    
    fig.set_size_inches(10, 5)
    
    ax[0].plot(mse_t, c="tab:orange", label="train")
    ax[0].plot(mse_v, c="tab:green", label="validation")
    ax[0].plot(range(smooth // 2, len(mse_t) - smooth // 2), mse_t_s, c="k")
    ax[0].plot(range(smooth // 2, len(mse_v) - smooth // 2), mse_v_s, c="k")
    ax[0].legend()
    ax[0].set_xlabel("epoch", fontsize=24)
    ax[0].set_ylabel("MSE", fontsize=24)
    ax[0].tick_params(axis="both", which="major", labelsize=16)

    ax[1].loglog(mse_t, c="tab:orange", label="train")
    ax[1].loglog(mse_v, c="tab:green", label="validation")
    ax[1].loglog(range(smooth // 2, len(mse_v) - smooth // 2), mse_t_s, c="k")
    ax[1].loglog(range(smooth // 2, len(mse_v) - smooth // 2), mse_v_s, c="k")
    ax[1].legend()
    ax[1].set_xlabel("epoch", fontsize=24)
    ax[1].set_ylabel("MSE", fontsize=24)
    ax[1].tick_params(axis="both", which="major", labelsize=16)
    
    plt.tight_layout()
    plt.show()