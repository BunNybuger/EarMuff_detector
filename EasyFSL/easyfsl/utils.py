"""
General utilities
"""

from typing import List, Tuple

import torchvision
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision import transforms


def plot_images(images: torch.Tensor, title: str, images_per_row: int):
    """
    Plot images in a grid.
    Args:
        images: 4D mini-batch Tensor of shape (B x C x H x W)
        title: title of the figure to plot
        images_per_row: number of images in each row of the grid
    """
    # import torchvision.transforms
    # transform_tensor_to_img = torchvision.transforms.ToPILImage()
    #inverse normalize
    #NORMALIZE_DEFAULT = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

    # inv_tensor = invTrans(constant_support_images[9,:,:,:])
    # my_img = transform_tensor_to_img(inv_tensor)
    # my_img.show()
    plt.figure(figsize=(20, 20))
    plt.axis('off')
    plt.title(title)
    plt.imshow(
        torchvision.utils.make_grid(invTrans(images), nrow=images_per_row).permute(1, 2, 0)
    )


def sliding_average(value_list: List[float], window: int) -> float:
    """
    Computes the average of the latest instances in a list
    Args:
        value_list: input list of floats (can't be empty)
        window: number of instances to take into account. If value is 0 or greater than
            the length of value_list, all instances will be taken into account.

    Returns:
        average of the last window instances in value_list

    Raises:
        ValueError: if the input list is empty
    """
    if len(value_list) == 0:
        raise ValueError("Cannot perform sliding average on an empty list.")
    return np.asarray(value_list[-window:]).mean()


def compute_backbone_output_shape(backbone: nn.Module) -> Tuple[int]:
    """
    Compute the dimension of the feature space defined by a feature extractor.
    Args:
        backbone: feature extractor

    Returns:
        shape of the feature vector computed by the feature extractor for an instance

    """
    input_images = torch.ones((4, 3, 32, 32))
    output = backbone(input_images)

    return tuple(output.shape[1:])


def compute_prototypes(
    support_features: torch.Tensor, support_labels: torch.Tensor
) -> torch.Tensor:
    """
    Compute class prototypes from support features and labels
    Args:
        support_features: for each instance in the support set, its feature vector
        support_labels: for each instance in the support set, its label

    Returns:
        for each label of the support set, the average feature vector of instances with this label
    """

    n_way = len(torch.unique(support_labels))
    # Prototype i is the mean of all instances of features corresponding to labels == i
    return torch.cat(
        [
            support_features[torch.nonzero(support_labels == label)].mean(0)
            for label in range(n_way)
        ]
    )
