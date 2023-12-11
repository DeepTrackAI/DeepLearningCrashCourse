#from torch.utils.data import Dataset
#import torch
#import os
#import numpy as np
import matplotlib.pyplot as plt
import torch.nn


def get_label(image):
    from numpy import array

    position = array(image.get_property("position"))
    return position


class ParticleDatasetSimul(Dataset):
    def __init__(self, pipeline, data_size):
        im = [pipeline.update().resolve() for _ in range(data_size)]
        self.pos = np.array([get_label(image) for image in im])[:, [1, 0]]
        self.im = np.array(im).squeeze()

    def __len__(self):
        return self.im.shape[0]

    def __getitem__(self, idx):
        img = torch.tensor(self.im[idx, np.newaxis, :, :]).float()
        labels = torch.tensor(self.pos[idx] / img.shape[-1] - 0.5).float()
        sample = [img, labels]
        return sample
