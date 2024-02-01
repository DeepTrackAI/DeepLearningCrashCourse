"""Module for the Denoising Autoencoder Example."""
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset


def plot_image(title, image):
    """Plot a grayscale image with a title."""
    
    plt.imshow(image, cmap="gray")
    plt.title(title, fontsize=24)
    plt.axis("off")
    plt.show()


class SimulatedDataset(Dataset):
    """Simulated dataset generating pairs of noisy and clean images."""
    
    def __init__(self, pipeline, buffer_size, replace=0):
        """Initialize the dataset."""
        
        self.buffer_size = buffer_size
        self.pipeline = pipeline
        self.replace = replace
        self.images = [pipeline.update().resolve() for _ in range(buffer_size)]

    def __len__(self):
        """Return the size of the dataset buffer."""
        
        return self.buffer_size

    def __getitem__(self, idx):
        """Retrieve a noisy-clean image pair from the dataset."""
        
        if np.random.rand() < self.replace:
            self.images[idx] = self.pipeline.update().resolve()
            
        image_pair = self.images[idx]
        noisy_image, clean_image = image_pair[0], image_pair[1]
        
        return noisy_image, clean_image
