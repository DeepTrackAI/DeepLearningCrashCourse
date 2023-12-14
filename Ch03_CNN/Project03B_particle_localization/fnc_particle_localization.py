import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import numpy as np
import torch
from torch.utils.data import Dataset


def load_video(path, frames_to_load, image_size):
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


class ManualAnnotation:
    def __init__(self, images):
        self.images = images
        self.positions = []
        self.i = 0
        self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5))
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False

    def start(self):
        self.im = self.ax.imshow(self.images[self.i], cmap="gray", 
                                 vmin=0, vmax=1)
        self.text = self.ax.text(3, 5, 
                                 f"Frame {self.i + 1} of {len(self.images)}", 
                                 color="white", fontsize=12)
        self.ax.axis("off")
        self.cursor = Cursor(self.ax, useblit=True, color="red", linewidth=1)
        self.cid = self.fig.canvas.mpl_connect("button_press_event", 
                                               self.onclick)
        self.next_image()
        plt.show()

    def next_image(self):
        self.im.set_data(self.images[self.i])
        self.text.set_text(f"Frame {self.i + 1} of {len(self.images)}")
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


class AnnotatedDataset(Dataset):
    def __init__(self, file_images, file_positions):
        self.images = np.load(file_images)
        self.positions = np.load(file_positions)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        im = torch.tensor(self.images[idx, np.newaxis, :, :]).float()
        pos = torch.tensor(self.positions[idx] / im.shape[-1] - .5).float()
        sample = [im, pos]
        return sample


def plot_simulated_particles(image_pipeline):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 4, figsize=(25, 12))
    for i, ax in enumerate(axes.flatten()):
        output_image = image_pipeline.update().resolve()
        ax.imshow(np.squeeze(output_image), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.show()


def get_label(image):
    from numpy import array

    position = array(image.get_property("position"))
    
    return position


def plot_simulated_particles_with_positions(image_pipeline):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 4, figsize=(25, 12))
    for i, ax in enumerate(axes.flatten()):
        output_image = image_pipeline.update().resolve()
        particle_position = get_label(output_image)

        ax.imshow(np.squeeze(output_image), cmap="gray")
        ax.scatter(particle_position[1], particle_position[0], s=60,
                   facecolors="none", edgecolor="g", linewidth=4)
        
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


class SimulatedDataset(Dataset):
    def __init__(self, pipeline, data_size):
        images = [pipeline.update().resolve() for _ in range(data_size)]
        self.images = np.array(images).squeeze()
        self.positions = np.array([get_label(im) for im in images])[:, [1, 0]]

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, idx):
        im = torch.tensor(self.images[idx, np.newaxis, :, :]).float()
        pos = torch.tensor(self.positions[idx] / image.shape[-1] - 0.5).float()
        sample = [im, pos]
        return sample
