import numpy as np
import pandas as pd

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

def draw_steering_distributions(steerings, ratio=0.5):
    """ Visualize steering distribution
    """
    # Raw steering angles:
    df_steering_dist = pd.DataFrame(
        data = steerings,
        columns = ['steering']
    )
    # Classes:
    df_steering_dist['class'] = np.round(
        (np.absolute(df_steering_dist['steering']) + 0.1)/0.1
    ).astype(np.int)
    # Straight line weight
    df_steering_dist['straight_weight'] = df_steering_dist['class']
    # Curved line weight
    df_steering_dist['curved_weight'] = df_steering_dist['class'].apply(lambda x: x**2)
    # Sample weight:
    df_steering_dist['sample_weight'] = (
        (1.0 - ratio)*df_steering_dist['straight_weight'] + ratio*df_steering_dist['curved_weight']
    )
    # Summary
    df_steering_dist = df_steering_dist.groupby('class').sum()

    classes = df_steering_dist.index.values
    width = 0.40

    fig, ax = plt.subplots(figsize=(16,9))
    legend_original = ax.bar(
        classes - 0.5*width,
        df_steering_dist['straight_weight'],
        width, color='r'
    )
    legend_balanced = ax.bar(
        classes + 0.5*width,
        df_steering_dist['sample_weight'],
        width, color='g'
    )

    # add some text for labels, title and axes ticks
    ax.set_xlabel('Steering')
    ax.set_ylabel('Weight')
    ax.set_title('Steering Distribution')
    ax.set_xticks(classes)

    ax.legend(
        (legend_original[0], legend_balanced[0]),
        ('Original', 'Balanced')
    )

    plt.show()
