# Set up session:
import argparse

from os.path import join

import numpy as np
import pandas as pd
import cv2

class Dataset:
    """ Behavioral cloning dataset iterator
    """
    LOG_FILENAME = "driving_log.csv"

    def __init__(
        self,
        dataset_path,
        steering_correction = 0.25
    ):
        # Dataset log:
        self.dataset_path = dataset_path
        self.driving_log = pd.read_csv(
            join(dataset_path, Dataset.LOG_FILENAME)
        )

        # Steering angle correction for left & right camera images:
        self.correction = steering_correction

        # Dataset index:
        self.idx = 0
        self.N = len(self.driving_log)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx < self.N:
            # Load dataset:
            (center, left, right, steering) = [
                str(val).strip()
                for val in
                self.driving_log.ix[
                    self.idx,
                    ['center', 'left', 'right', 'steering']
                ].values
            ]

            # Format fields
            (center, left, right) = [
                join(self.dataset_path, filename)
                for filename in (center, left, right)
            ]
            steering = float(steering)

            # Read original images and augment them:
            original_images = [
                cv2.imread(filename)
                for filename in (center, left, right)
            ]
            flipped_images = [
                np.fliplr(image)
                for image in original_images
            ]
            original_steerings = [
                steering,
                steering + self.correction,
                steering - self.correction
            ]
            flipped_steerings = [
                -steering for steering in original_steerings
            ]

            images = np.array(original_images + flipped_images)
            steerings = np.array(original_steerings + flipped_steerings)

            # Update index:
            self.idx += 1

            return (images, steerings)
        else:
            raise StopIteration()

if __name__ == '__main__':
    # Parse command-line arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        required=True,
        help="Path to behavioral cloning dataset."
    )
    args = vars(parser.parse_args())

    # Use case 01 -- Dataset iterator:
    dataset = Dataset(args["dataset"])
    for (images, steerings) in iter(dataset):
        for image, steering in zip(images, steerings):
            cv2.imshow(
                "Steering: {:.2f}".format(steering),
                image
            )
            cv2.waitKey(0)
        break
