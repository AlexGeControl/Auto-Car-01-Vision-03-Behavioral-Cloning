# Set up session:
import numpy as np
import cv2

# Sklearn transformer interface:
from sklearn.base import TransformerMixin

class Preprocessor(TransformerMixin):
    def __init__(
        self,
        input_size = (320, 160),
        cropping = ((50, 20), (0, 0)),
        output_size = (64, 18)
    ):
        W, H = input_size
        (
            (TOP_OFFSET, BOTTOM_OFFSET),
            (LEFT_OFFSET, RIGHT_OFFSET)
        ) = cropping

        self.top = TOP_OFFSET
        self.bottom = H - BOTTOM_OFFSET
        self.left = LEFT_OFFSET
        self.right = W - RIGHT_OFFSET

        self.output_size = output_size

    def transform(self, X):
        """ Preprocess raw frames
        """
        return np.asarray(
            [self._resize(x) for x in X]
        )

    def fit(self, X, y=None):
        return self

    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)

    def _resize(self, X):
        """
        """
        # Crop:
        cropped = X[self.top:self.bottom, self.left:self.right]

        # Resize:
        resized = cv2.resize(cropped, self.output_size)

        return resized
