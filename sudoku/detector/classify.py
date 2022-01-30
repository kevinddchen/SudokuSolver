import os

import numpy as np
import torch
import torchvision.transforms.functional as TF

import sudoku.network

# ------------------------------------------------------------------------------

class DigitClassifier(object):

    '''
    Convenience class to perform evaluation on images.

    Args:
        path_to_weights (str): Pretrained network weights. If None, then will
            try to find the default weights included with this repository.
            Defaults to None.
    '''

    def __init__(self, path_to_weights: str = None):
        self.model = sudoku.network.get_model()
        if path_to_weights is None:
            path_to_weights = self._path_to_default_weights()
        sudoku.network.load_weights(self.model, path_to_weights)
        self.model.eval()

    def __call__(self, img: np.ndarray) -> int:

        '''
        Call to classify digit.

        Args:
            x (ndarray<uint8>): 28x28 grayscale image.

        Returns:
            (int): Digit in 1, 2, ..., 9.
        '''
        
        x = TF.to_tensor(img).unsqueeze(0)
        with torch.no_grad():
            out = self.model(x)
        pred = out[:, 1:].argmax(1)     ## Exclude 0
        return pred.item()+1

    def _path_to_default_weights(self) -> str:
        return os.path.join(sudoku.network.__path__[0], 'checkpoints/weights_epoch_06.pth')
