
"""Module for the LodeSTAR Example."""
import matplotlib.pyplot as plt


def plot_position_comparison(positions, predictions):
    """Plot comparison between predicted and real particle positions."""

    plt.figure(figsize=(14, 8))
    grid = plt.GridSpec(4, 7, wspace=.2, hspace=.1)

    plt.subplot(grid[1:, :3])
    plt.scatter(positions[:, 0], predictions[:, 0], alpha=.5) 
    plt.axline((25, 25), slope=1, color="black")
    plt.xlabel("True Horizontal Position")
    plt.ylabel("Predicted Horizontal Position")
    plt.axis("equal")    

    plt.subplot(grid[1:, 4:])
    plt.scatter(positions[:, 1], predictions[:, 1], alpha=.5)
    plt.axline((25, 25), slope=1, color="black")
    plt.xlabel("True Vertical Position")
    plt.ylabel("Predicted Vertical Position")
    plt.axis("equal")

    plt.show()