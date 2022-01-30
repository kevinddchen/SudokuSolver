import os

import torch
import torch.nn as nn

# ------------------------------------------------------------------------------

def get_model(L: int = 8, D: int = 256) -> nn.Module:

    '''
    Neural network model for digit classification. Inputs are Nx1x28x28 tensors
    of floats in the range [0, 1].

    Args:
        L (int): Number of latent features.
        D (int): Size of dense layers.

    Returns:
        (nn.Module): Neural network model.
    '''

    model = nn.Sequential(
        nn.Conv2d(  1,   L, 3, padding='same'),
        nn.ReLU(inplace=True),
        nn.Conv2d(  L,   L, 3, padding='same'),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(  L, 2*L, 3, padding='same'),
        nn.ReLU(inplace=True),
        nn.Conv2d(2*L, 2*L, 3, padding='same'),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(2*L, D, 7, padding=0),
        nn.ReLU(inplace=True),
        nn.Flatten(1),
        nn.Linear(D, D),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(D, 10)
    )
    return model

# ------------------------------------------------------------------------------

def init_weights(m: nn.Module):
    '''He initialization for weights.'''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

# ------------------------------------------------------------------------------

def checkpoint_path(checkpoint_dir: str, epoch: int) -> str:
    '''Path to checkpoint file.'''
    return os.path.join(checkpoint_dir, f'weights_epoch_{epoch:02d}.pth')

# ------------------------------------------------------------------------------

def save_weights(model: nn.Module, path: str):
    '''Save model weights to file.'''
    torch.save(model.state_dict(), path)

# ------------------------------------------------------------------------------

def load_weights(model: nn.Module, path: str):
    '''Load model weights from file.'''
    pretrained_dict = torch.load(path)
    model.load_state_dict(pretrained_dict)
    