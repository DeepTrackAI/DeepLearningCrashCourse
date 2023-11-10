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
    from numpy import array
    from sklearn.metrics import roc_curve, auc
    from matplotlib import pyplot as plt

    # im = []
    # gt = []
    # pred = []

    # for images,labels in loader:
    #     predictions = classifier(images)

    #     [im.append(i) for i in images.detach().numpy()]
    #     # [gt.append(i.argmax()) for i in labels.numpy().astype(int)]
    #     # [pred.append(i.argmax()) for i in predictions.detach().numpy()]
    #     [gt.append(i[0]) for i in labels.numpy().astype(int)]
    #     [pred.append(i[0]) for i in predictions.detach().numpy()]

    im, gt = zip(*dataset)
    pred = CNN(torch.tensor(torch.stack(im)))
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

    return array(im), array(gt), array(pred), roc_auc


def plot_failure(images, gt, pred, threshold = 0.5, num_of_plots = 5):
    from matplotlib import pyplot as plt    
    
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