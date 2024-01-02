import os
from torch.utils.data import Dataset
from PIL import Image


class Holo2BrightDataset(Dataset):
    """
    Custom Dataset class for loading Holo2Bright dataset. It will help in not loading the entire dataset into memory at once, using torch transforms for data augmentation and preprocessing, and creating mini-batches of the data.
    """

    def __init__(self, root, train=True, transform=None, dataset_size=None):
        self.root = root
        self.transform = transform
        self.dataset_size = dataset_size

        # Set the train and test dataset directories
        if train:
            self.holography_dir = os.path.join(
                root, "holo2bright", "train", "holography"
            )
            self.brightfield_dir = os.path.join(
                root, "holo2bright", "train", "brightfield"
            )

        else:
            self.holography_dir = os.path.join(
                root, "holo2bright", "test", "holography"
            )
            self.brightfield_dir = os.path.join(
                root, "holo2bright", "test", "brightfield"
            )

        # Get the list of all the images in the dataset directory
        self.holography_images = os.listdir(self.holography_dir)
        self.brightfield_images = os.listdir(self.brightfield_dir)

    def __len__(self):
        return (
            len(self.brightfield_images)
            if self.dataset_size is None
            else self.dataset_size
        )

    def __getitem__(self, index):
        # Read the image from the directory
        holography_image = Image.open(
            os.path.join(self.holography_dir, self.holography_images[index])
        )
        brightfield_image = Image.open(
            os.path.join(self.brightfield_dir, self.brightfield_images[index])
        )

        # Apply transformations on the images
        if self.transform:
            holography_image = self.transform[0](holography_image)
            brightfield_image = self.transform[1](brightfield_image)

        return holography_image, brightfield_image
