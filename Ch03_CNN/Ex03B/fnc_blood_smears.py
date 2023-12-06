def plot_blood_smears(title, files):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axs.ravel()):
        image = plt.imread(files[i])
        ax.imshow(image)
        
    fig.suptitle(title, fontsize=16)
    plt.show()


def plot_roc(classifier, dataset):
    from torch import tensor, stack
    from sklearn.metrics import roc_curve, auc
    from matplotlib import pyplot as plt

    # calculate predictions
    images, gt = zip(*dataset)
    pred = classifier(tensor(stack(images))).tolist()
    
    # calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(gt, pred, pos_label=1) 
    roc = auc(fpr, tpr) 

    # plot the ROC curve
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc:.3f})", linewidth=2)
    plt.title("ROC curve")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.axis("square")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc = 'center right')
    plt.show()

    return images, gt, pred, roc


def plot_failures(images, gt, pred, threshold=.5, plot_num=5):
    from matplotlib import pyplot as plt 
    from numpy import array, squeeze
    
    pred = array(pred).squeeze()
    gt = array(gt).squeeze()
    images = array(images)

    false_positives = (pred > threshold) & (gt == 0)
    false_positives_images = images[false_positives]

    false_negatives = (pred < threshold) & (gt == 1)
    false_negatives_images = images[false_negatives]

    plt.figure(figsize=(plot_num * 2, 5))
    for i in range(plot_num):
        # false positives
        plt.subplot(2, plot_num, i + 1)
        plt.imshow(false_positives_images[i].transpose(1, 2, 0))
        if i == 0:
            plt.title("False positives", fontsize=16, y=1.1)

        # false negatives
        plt.subplot(2, plot_num, plot_num + i + 1)
        plt.imshow(false_negatives_images[i].transpose(1, 2, 0))
        if i == 0:
            plt.title("False negatives", fontsize=16, y=1.1)

    plt.tight_layout()
    plt.show()


class fwd_hook():
    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_func)

    def hook_func(self, layer, i, o):
        print("Forward hook running ...") 
        self.activations = o.detach().clone()
        print(f"Activations size: {self.activations.size()}")

    def __enter__(self, *args): 
        return self
    
    def __exit__(self, *args): 
        self.hook.remove()


class bwd_hook():
    def __init__(self, layer):
        self.hook = layer.register_full_backward_hook(self.hook_func)

    def hook_func(self, layer, gi, go):
        print("Backward hook running ...")
        self.gradients = go[0].detach().clone()
        print(f"Gradients size: {self.gradients.size()}")

    def __enter__(self, *args): 
        return self
    
    def __exit__(self, *args): 
        self.hook.remove()


def plot_activations(activations, cols=8):
    from matplotlib import pyplot as plt

    rows = -(activations.shape[0] // -cols)

    fig, axs = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    for i, ax in enumerate(axs.ravel()):
        ax.axis('off')
        if i < activations.shape[0]:
            ax.imshow(activations[i].numpy())
            ax.set_title(i)

    fig.tight_layout()
    plt.show()


def plot_gradcam(image, grad_cam):
    from matplotlib import pyplot as plt
    import skimage

    grad_cam = skimage.transform.resize(grad_cam, image.shape, order=2)
    grad_cam = skimage.exposure.rescale_intensity(grad_cam, out_range=(0.25, 1))

    plt.figure(figsize=(12, 5)) 

    plt.subplot(1, 3, 1)
    plt.imshow(image, interpolation="bilinear")
    plt.title("Original image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(grad_cam.mean(axis=-1), interpolation="bilinear")
    plt.title("Grad-CAM")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(image * grad_cam)
    plt.title("Overlay")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()