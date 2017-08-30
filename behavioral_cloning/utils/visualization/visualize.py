import numpy as np

import matplotlib.pyplot as plt

def draw_image_mosaic(images, size):
    """ Draw image mosaic consists of given image samples
    """
    # Initialize canvas:
    plt.figure(figsize=(size,size))
    # Turn off axes:
    plt.gca().set_axis_off()

    N = len(images)
    image_mosaic = np.vstack(
        [np.hstack(images[np.random.choice(N, size)]) for _ in range(size)]
    )

    if image_mosaic.shape[-1] == 1:
        H, W, _ = image_mosaic.shape
        image_mosaic.reshape((H, W))
        image_mosaic = np.dstack(
            tuple([image_mosaic] * 3)
        )

    plt.imshow(image_mosaic)
    plt.show()

def draw_label_distributions(class_counts, num_classes):
    """ Visualize label distributions by subsets
    """
    # Parse label counts:
    class_counts = class_counts / np.sum(class_counts, axis = 1, keepdims=True)
    (counts_train, counts_valid, counts_test) = class_counts

    classes = np.arange(num_classes)

    width = 0.30

    fig, ax = plt.subplots(figsize=(16,9))
    legend_train = ax.bar(
        classes - 1.5*width,
        counts_train,
        width, color='r'
    )
    legend_valid = ax.bar(
        classes - 0.5*width,
        counts_valid,
        width, color='g'
    )
    legend_test = ax.bar(
        classes + 0.5*width,
        counts_test,
        width, color='b'
    )

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Percentange')
    ax.set_title('Percentage by Subset')
    ax.set_xticks(classes)

    ax.legend(
        (legend_train[0], legend_valid[0], legend_test[0]),
        ('Train', 'Dev', 'Test')
    )

    plt.show()

def draw_top_k(image, labels, probs):
    """ Visualize top K predictions
    """
    fig, axes = plt.subplots(1, 2)

    # Parse data:
    N = len(labels)
    label_pos = np.arange(N)

    # Top K predictions:
    axes[0].barh(
        label_pos, probs,
        color='green'
    )
    axes[0].set_yticks(label_pos)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()  # labels read top-to-bottom
    axes[0].set_xlabel('Probability')
    axes[0].set_title("Top {}".format(N))

    # Input image:
    axes[1].imshow(image)
    axes[1].set_axis_off()
    axes[1].set_title("Web Image")

    plt.show()
