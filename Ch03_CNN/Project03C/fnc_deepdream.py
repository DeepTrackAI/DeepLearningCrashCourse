class fwd_hooks():
    def __init__(self, layers):
        self.layers = layers
        self.hook = []
        self.activations = []
        for layer in self.layers:
            self.hook.append(layer.register_forward_hook(self.hook_func))

    def hook_func(self, layer, input, output):
        self.activations.append(output)

    def __enter__(self, *args): 
         return self
    
    def __exit__(self, *args): 
        for h in self.hook:    
            h.remove()


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
    im_array = np.clip(im_array.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(im_array, 'RGB')