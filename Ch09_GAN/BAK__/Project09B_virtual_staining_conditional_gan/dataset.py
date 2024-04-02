from torchvision import transforms
import glob
import numpy as np
import os
from PIL import Image
import torch

from torch.utils.data import Dataset
from tqdm import tqdm


class VirtualStainingDataset(Dataset):
    _cache = {}  # Static variable for caching loaded images

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        normalize=[
            transforms.Normalize(mean=[0] * 13, std=[1] * 13),  # Input
            transforms.Normalize(mean=[0] * 3, std=[1] * 3),  # Target
        ],
        preload=False,
    ):
        self.transform = transform
        self.preload = preload
        self.normalize = normalize
        self.images = []

        if train:
            self.image_dir = os.path.join(root, "train", "scott_1_0")
        else:
            self.image_dir = os.path.join(root, "test", "scott_1_0")

        pattern = "lab-Rubin,condition-scott_1_0,acquisition_date,year-2016,month-2,day-6,well-r0*c0*,depth_computation,value-MAXPROJECT,is_mask-false,kind,value-ORIGINAL.png"

        self.image_list = glob.glob(os.path.join(self.image_dir, pattern))

        self.cache_key = self.image_dir

        if self.preload:
            # Check if the images are already cached
            if self.cache_key in VirtualStainingDataset._cache:
                self.images = VirtualStainingDataset._cache[self.cache_key]
            else:
                # Preload all images and cache them
                for image_path in tqdm(
                    self.image_list,
                    total=len(self.image_list),
                    desc="Preloading images",
                ):
                    self.images.append(self.load_image(image_path))
                VirtualStainingDataset._cache[self.cache_key] = self.images

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if self.preload:
            input_image, target_image = self.images[idx]
        else:
            input_image, target_image = self.load_image(self.image_list[idx])

        if self.transform:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            input_image = self.transform(input_image)

            torch.manual_seed(seed)
            target_image = self.transform(target_image)

        input_image = self.normalize[0](input_image)
        target_image = self.normalize[1](target_image)

        return input_image, target_image

    def load_image(self, image_path):
        # Load the target image
        target_image = np.array(Image.open(image_path))

        # Load the input image
        input_image = []
        for i in range(0, 13):
            img_path = image_path.replace(
                "depth_computation", "z_depth-{},channel".format(i)
            )
            img_path = img_path.replace("value-MAXPROJECT", "value-BRIGHTFIELD")

            input_img_single_depth = np.array(Image.open(img_path).convert("L"))
            input_image.append(input_img_single_depth)

        input_image = np.stack(input_image, axis=-1)

        return input_image, target_image
