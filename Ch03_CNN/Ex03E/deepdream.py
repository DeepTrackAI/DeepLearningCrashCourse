class fwd_hook():
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_func)   
    def hook_func(self, m, i, o):
        #print('Forward hook running...') 
        self.stored = o#.detach().clone()
        #print(f'Activations size: {self.stored.size()}')
    def __enter__(self, *args): 
        return self
    def __exit__(self, *args): 
        self.hook.remove()

def preprocess(image):
    import torch 
    from torchvision import transforms
    normalize = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
                            ])
    # tensor = normalize(torch.from_numpy(image.transpose(2,0,1)).type(torch.FloatTensor).unsqueeze(0)).requires_grad_(True)
    tensor = normalize(image).unsqueeze(0).requires_grad_(True)
    return tensor

def deprocess(tensor):
    import numpy as np
    from torchvision import transforms
    denormalize = transforms.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                                    std = [1/0.229, 1/0.224, 1/0.225] )
    im = np.array(denormalize(tensor.detach().squeeze())).transpose(1,2,0)
    return im





def plot_examples(uninfected_files,infected_files):

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 8))
    subfigs = fig.subfigures(1, 2)
    subfigs[0].suptitle('Uninfected')
    subfigs[1].suptitle('Infected')

    axsLeft = subfigs[0].subplots(4, 4)
    axsRight = subfigs[1].subplots(4, 4)

    for i, (ax, bx) in enumerate(zip(axsLeft.reshape(-1), axsRight.reshape(-1)) ):
        imageLeft = plt.imread(uninfected_files[i])
        ax.imshow(imageLeft)
        imageRight = plt.imread(infected_files[i])
        bx.imshow(imageRight)

    plt.show()

def plot_ROC_AUC(classifier, dataset):
    from torch import tensor, stack
    from sklearn.metrics import roc_curve, auc
    from matplotlib import pyplot as plt

    im, gt = zip(*dataset)
    pred = classifier(tensor(stack(im))).tolist()
    # calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(gt, pred, pos_label=1) 
    roc_auc = auc(fpr, tpr) 

    # plot the ROC curve
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", linewidth=2)
    plt.title("ROC curve")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.axis("square")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc = 'center right')
    plt.show()

    return im, gt, pred, roc_auc


def plot_failure(images, gt, pred, threshold = 0.5, num_of_plots = 5):
    from matplotlib import pyplot as plt 
    from numpy import array, squeeze   
    
    pred = array(pred).squeeze()
    gt = array(gt).squeeze()
    images = array(images)

    pred_class = pred > threshold

    false_positives = (pred_class == 1) & (gt == 0)
    false_positives_images = images[false_positives]

    false_negatives = (pred_class == 0) & (gt == 1)
    false_negatives_images = images[false_negatives]

    plt.figure(figsize=(num_of_plots*2, 5))
    for i in range(num_of_plots):

        # false positives
        plt.subplot(2, num_of_plots, i + 1)
        plt.imshow(false_positives_images[i].transpose(1, 2, 0))
        if i == 0:
            plt.title("False positives", fontsize=16, y=1.1)

        # false negatives
        plt.subplot(2, num_of_plots, i + num_of_plots + 1)
        plt.imshow(false_negatives_images[i].transpose(1, 2, 0))
        if i == 0:
            plt.title("False negatives", fontsize=16, y=1.1)

    plt.tight_layout()
    plt.show()


def plot_filters_activations(input,n_rows, label = '',normalize = True):
    from matplotlib import pyplot as plt
    fig,axes = plt.subplots(n_rows, -(input.shape[0] // -n_rows), figsize=(2*(-(input.shape[0] // -n_rows)),2*n_rows))
    for i in range(-(input.shape[0] // -n_rows)*n_rows):
        try: 
            p  = input[i].permute(1,2,0).numpy()
            if normalize: 
                p -= p.min(axis=(0,1), keepdims=True)
                p /= p.max(axis=(0,1), keepdims=True)
            axes.ravel()[i].axis('off')
            axes.ravel()[i].imshow(p)
            axes.ravel()[i].set_title(i)
        except: 
            axes.ravel()[i].axis('off')
    fig.suptitle(label, fontsize = 16)
    fig.tight_layout()
    plt.show()


def plot_gradcam(gcam, img):
    from matplotlib import pyplot as plt
    from skimage.transform import resize
    from skimage.exposure import rescale_intensity

    gcam = resize(gcam, img.shape, order = 2)
    gcam = rescale_intensity(gcam,out_range=(0.25,1))

    plt.figure(figsize=(12, 5)) 
    plt.subplot(1, 3, 1)
    plt.imshow(img, interpolation = 'bilinear')
    plt.title('Original image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(gcam.mean(axis=-1), interpolation = 'bilinear')
    plt.title('Grad-CAM')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(img*gcam)
    plt.title('Overlay')
    plt.axis('off')
    plt.tight_layout()
    plt.show()



class bwd_hook():
    def __init__(self, m):
        self.hook = m.register_full_backward_hook(self.hook_func)
    def hook_func(self, m, gi, go):
        print('Backward hook running...')
        self.stored = go[0].detach().clone()
        print(f'Gradients size: {self.stored.size()}')
    def __enter__(self, *args): 
        return self
    def __exit__(self, *args): 
        self.hook.remove()