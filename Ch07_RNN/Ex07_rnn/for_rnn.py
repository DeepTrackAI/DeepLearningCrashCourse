"""Module for the RNN Example."""

import matplotlib.pyplot as plt
import numpy as np


def plot_data(data, header, start=0, samples_per_cycle=144, cycles=14):
    """Plot data highlighting periodic cycles."""

    fig, axes = plt.subplots(7, 2, figsize=(16, 12), sharex=True)

    for i, ax in enumerate(axes.flatten()):
        ax.plot(
            np.arange(start, start + samples_per_cycle * cycles),
            data[start : start + samples_per_cycle * cycles, i],
            label=header[i],
        )
        ax.legend()
        ax.set_xlim(start, start + samples_per_cycle * cycles)

        for cycle in range(1, cycles):
            ax.axvline(
                x=start + cycle * samples_per_cycle,
                color="gray",
                linestyle="--",
                linewidth=0.5,
            )

    plt.tight_layout()
    plt.show()


def plot_training(epochs, train_losses, val_losses, benchmark):
    """Plot the training and validation losses."""

    plt.plot(range(epochs), train_losses, label="Training Loss")
    plt.plot(range(epochs), val_losses, label="Validation Loss")
    plt.plot(
        [0, epochs - 1],
        [benchmark, benchmark],
        linestyle="--",
        color="k",
        label="Benchmark",
    )
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim([0, epochs - 1])
    plt.show()
