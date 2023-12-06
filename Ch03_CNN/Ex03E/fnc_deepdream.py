class fwd_hooks():
    def __init__(self, layers):
        self.layers = layers
        self.hook = []
        self.activations = []
        for layer in self.layers:
            self.hook.append(layer.register_forward_hook(self.hook_func))

    def hook_func(self, m, i, o):
        self.activations.append(o)

    def __enter__(self, *args): 
         return self
    
    def __exit__(self, *args): 
        for h in self.hook:    
            h.remove() 


def plot_deepdream(im, im_out):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(im)
    plt.title('Original image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(im_out)
    plt.title('Deepdream image') 
    plt.axis('off')
    
    plt.show()