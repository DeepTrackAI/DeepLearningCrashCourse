from torch.utils.data import Dataset
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn
from matplotlib.widgets import Cursor


def load_video(path, frames_to_load=100, image_size=51):
    import cv2
    import numpy as np

    video = cv2.VideoCapture(path)
    data = []
    for _ in range(frames_to_load):
        (_, frame) = video.read()
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255
        frame = cv2.resize(frame, (image_size, image_size))
        data.append(frame)
    return np.array(data)


class ParticleCenter:
    def __init__(self, images):
        self.images = images
        self.positions = []
        self.i = 0
        self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5))
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False

    def start(self):
        self.im = self.ax.imshow(self.images[self.i], cmap="gray", vmin=0, vmax=1)
        self.text = self.ax.text(
            3,
            5,
            "Frame " + str(self.i + 1) + " of " + str(len(self.images)),
            color="white",
            fontsize=12,
        )
        self.ax.axis("off")
        self.cursor = Cursor(self.ax, useblit=True, color="red", linewidth=1)
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.next_image()
        plt.show()

    def next_image(self):
        im = self.images[self.i]
        self.im.set_data(im)
        self.text.set_text("Frame " + str(self.i + 1) + " of " + str(len(self.images)))
        self.fig.canvas.draw_idle()

    def onclick(self, event):
        self.positions.append([event.xdata, event.ydata])
        if self.i < len(self.images) - 1:
            self.i += 1
            self.next_image()
        else:
            self.fig.canvas.mpl_disconnect(self.cid)
            plt.close()
            return


class ParticleDataset(Dataset):
    def __init__(self, file, dir):
        self.im = np.load(os.path.join(dir, file))
        name, ext = os.path.splitext(file)
        self.pos = np.load(os.path.join(dir, name + "_labels" + ext))

    def __len__(self):
        return self.im.shape[0]

    def __getitem__(self, idx):
        img = torch.tensor(self.im[idx, np.newaxis, :, :]).float()
        labels = torch.tensor(self.pos[idx] / img.shape[-1] - 0.5).float()
        sample = [img, labels]
        return sample


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


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
