

def plot_training_metrics(metrics):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2)

    axes[0].plot(metrics["train_loss_epoch"], label="Train Loss")
    axes[0].plot(metrics["val_loss_epoch"], label="Validation Loss")
    axes[0].set_xticks([])
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(metrics["trainArgmaxJaccardIndex_epoch"], label="Train Jaccard Index")
    axes[1].plot(metrics["valArgmaxJaccardIndex_epoch"], label="Validation Jaccard Index")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Jaccard Index")
    axes[1].legend()

    plt.tight_layout()
    plt.show()