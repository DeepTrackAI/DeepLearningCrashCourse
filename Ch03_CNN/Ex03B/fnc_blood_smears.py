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