def image_to_tensor(im, mean, std):
    import torchvision.transforms as tt

    normalize = tt.Compose([tt.ToTensor(), tt.Normalize(mean, std)])

    return normalize(im).unsqueeze(0).requires_grad_(True)


def tensor_to_image(image, mean, std):
    import torchvision.transforms as tt
    import numpy as np
    from PIL import Image

    denormalize = tt.Normalize(mean=-mean / std, std=1 / std)

    im_array = denormalize(image.data.clone().detach().squeeze()).numpy()
    im_array = np.clip(im_array.transpose(1, 2, 0) * 255, 0, 255)
    im_array = im_array.astype(np.uint8)
    return Image.fromarray(im_array, "RGB")


class Fwd_Hooks():
    def __init__(self, layers):
        self.hooks = []
        self.activations_list = []
        for layer in layers:
            self.hooks.append(layer.register_forward_hook(self.hook_func))

    def hook_func(self, layer, input, output):
        self.activations_list.append(output)

    def __enter__(self, *args): 
        return self
    
    def __exit__(self, *args): 
        for hook in self.hooks:
            hook.remove()