from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from utils import torch_to_numpy,clip_percentile


def clip_percentile(array, percentile=0.1):
    """
    Clip the values of an array at a specified percentile.

    Args:
        array (np.ndarray): The input array to be clipped.
        percentile (float, optional): The percentile value for clipping. Defaults to 0.1.

    Returns:
        np.ndarray: The clipped array.
    """
    return np.clip(array, None, np.percentile(array, 100 - percentile))

def plot_alpha(image_tensor, alpha_tensor, percentile=10, save_output=False, display_output=True, blur_strength=2, cropping_area=None, figure_size=(10, 10)):
    """
    Remove outliers and plot images with transparency (alpha) channels.

    Args:
        image_tensor (torch.Tensor): Tensor of images to be processed and displayed. 
                                     Shape should be (N, C, H, W) or (C, H, W) where N is the number of images,
                                     C is the number of channels, H is height, and W is width.
        alpha_tensor (torch.Tensor): Tensor of alpha (transparency) channels corresponding to the images.
                                     Shape should match image_tensor.
        percentile (int, optional): Percentile value for clipping to remove outliers in the alpha channel. Defaults to 10.
        save_output (bool, optional): Whether to save the output images as files. Defaults to False.
        display_output (bool, optional): Whether to display the output images. Defaults to True.
        blur_strength (float, optional): Sigma value for Gaussian blur applied to the alpha channel. Defaults to 2.
        cropping_area (tuple, optional): Tuple specifying the cropping area in the format ((y1, y2), (x1, x2)). 
                                         If None, no cropping is applied. Defaults to None.
        figure_size (tuple, optional): Size of the figure for displaying images. Defaults to (10, 10).

    Returns:
        None
    """
    image_array = torch_to_numpy(image_tensor)
    alpha_array = torch_to_numpy(alpha_tensor)

    # Handle single image
    if len(image_array.shape) == 3:
        image_array = image_array[np.newaxis, ...]
        alpha_array = alpha_array[np.newaxis, ...]

    num_images = image_array.shape[0]
    figure, axis_list = plt.subplots(1, num_images, figsize=(figure_size[0] * num_images, figure_size[1]))

    if num_images == 1:
        axis_list = [axis_list]  # Make it iterable if there's only one subplot

    for idx in range(num_images):
        current_image = image_array[idx]
        current_alpha = alpha_array[idx]

        if current_alpha.shape[0] == 1:
            current_alpha = np.moveaxis(current_alpha, 0, -1)

        # Normalize image
        current_image -= current_image.mean()
        current_image /= current_image.std()
        current_image -= current_image.min()
        current_image /= current_image.max()

        # Process alpha channel
        current_alpha = np.mean(np.array(current_alpha).copy(), -1, keepdims=True)
        current_alpha = clip_percentile(current_alpha, percentile)
        current_alpha = current_alpha / current_alpha.max()

        # Blur alpha channel
        current_alpha = current_alpha.squeeze()
        current_alpha = gaussian_filter(current_alpha, sigma=blur_strength)
        current_alpha = current_alpha[:, :, np.newaxis]

        if cropping_area is None:
            axis_list[idx].imshow(np.concatenate([current_image, current_alpha], -1))
        else:
            cropped_image = current_image[cropping_area[0][0]:cropping_area[0][1], cropping_area[1][0]:cropping_area[1][1]]
            cropped_alpha = current_alpha[cropping_area[0][0]:cropping_area[0][1], cropping_area[1][0]:cropping_area[1][1]]
            axis_list[idx].imshow(np.concatenate([cropped_image, cropped_alpha], -1))

        axis_list[idx].axis('off')

        if save_output:
            plt.savefig(f'output_image_{idx}.png', bbox_inches='tight')

    if display_output:
        plt.show()
    plt.close()


def plot(image_tensor, clip_percent=0.1, normalize=True):
    """
    Remove outliers and plot image(s).

    Args:
        image_tensor (torch.Tensor): Tensor of image(s) to be processed and displayed. 
                                     Shape can be (N, C, H, W) or (C, H, W) where N is the number of images,
                                     C is the number of channels, H is height, and W is width.
        clip_percent (float, optional): Percentile value for clipping to remove outliers. If None, no clipping is applied. Defaults to 0.1.
        normalize (bool, optional): Whether to normalize the images. Defaults to True.

    Returns:
        None
    """
    image_array = torch_to_numpy(image_tensor)
    
    if len(image_array.shape) == 3:
        if clip_percent is not None:
            image_array = clip_percentile(image_array, clip_percent)
        if normalize:
            image_array -= image_array.mean()
            image_array /= image_array.std()
            image_array -= image_array.min()
            image_array /= image_array.max()
        plt.imshow(image_array)
        plt.axis('off')
    else:
        num_images = image_array.shape[0]
        for idx in range(num_images):
            plt.subplot(1, num_images, idx + 1)
            current_image = image_array[idx]
            if clip_percent is not None:
                current_image = clip_percentile(current_image, clip_percent)
            if normalize:
                current_image -= current_image.mean()
                current_image /= current_image.std()
                current_image -= current_image.min()
                current_image /= current_image.max()

            plt.imshow(current_image)
            plt.axis('off')

        plt.show()