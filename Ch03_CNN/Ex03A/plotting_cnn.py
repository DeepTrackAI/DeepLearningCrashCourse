def plot_image(image):
    import matplotlib.pyplot as plt
    
    plt.imshow(image, cmap="gray", aspect="equal", 
               extent=[0, image.shape[1], 0, image.shape[0]])
    plt.grid(color="red", linewidth=1)
    plt.xticks(range(0, image.shape[1] + 1))
    plt.xlim(0, image.shape[1])
    plt.yticks(range(0, image.shape[0] + 1))
    plt.ylim(0, image.shape[0])
    plt.tight_layout()
    plt.show()