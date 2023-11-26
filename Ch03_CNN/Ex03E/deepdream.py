class fwd_hooks():
    def __init__(self, ms):
        self.ms = ms
        self.hook, self.stored = [], []
        for m in self.ms:
            self.hook.append(m.register_forward_hook(self.hook_func))
    def hook_func(self, m, i, o):
        # print('Forward hook running...') 
        self.stored.append(o)
        # print(f'Activations size: {o.size()}')
    def __enter__(self, *args): 
         return self
    def __exit__(self, *args): 
        for h in self.hook:    
            h.remove() 


def preprocess(image, mean_ds, std_ds):
    import torch 
    from torchvision import transforms
    normalize = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(mean=mean_ds,
                                 std=std_ds),
                            ])
    # tensor = normalize(torch.from_numpy(image.transpose(2,0,1)).type(torch.FloatTensor).unsqueeze(0)).requires_grad_(True)
    tensor = normalize(image).unsqueeze(0).requires_grad_(True)
    return tensor

def deprocess(tensor, mean_ds, std_ds):
    import numpy as np
    from torchvision import transforms
    denormalize = transforms.Normalize(mean = -mean_ds/std_ds, 
                                    std = 1/std_ds )
    im = np.array(denormalize(tensor.detach().squeeze())).transpose(1,2,0)
    return im

def plot_dream(im_in,im_out):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5)) 
    plt.subplot(1, 2, 1)
    plt.imshow(im_in)
    plt.title('Original image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(im_out)
    plt.title('Deepdream image') 
    plt.axis('off')
    plt.show()