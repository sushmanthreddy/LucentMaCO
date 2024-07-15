from __future__ import absolute_import, division, print_function

import os
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np


totensor = ToTensor()


default_model_input_size = (224,224)
default_img_size = (512,512) 
default_model_input_range = (-2,2) 


def img_to_img_tensor(image, target_size=default_img_size, crop=True):
    """
    Convert an image to a tensor and resize it to the specified size.

    Args:
        image (str or PIL.Image or torch.Tensor): The input image. Can be a file path, a PIL image, or a torch tensor.
        target_size (tuple, optional): The target size for resizing the image. Defaults to default_img_size.
        crop (bool, optional): Whether to crop the image before resizing. Defaults to True.

    Returns:
        torch.Tensor: The image converted to a tensor and resized to the target size.
    """
    if isinstance(image, str):
        image = Image.open(os.path.abspath(image))
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        if image.mode == 'L':
            image = image.convert('RGB')  # Convert grayscale to RGB

    if not isinstance(image, torch.Tensor): 
        image = totensor(image)

    if image.dim() == 3: 
        image = image.unsqueeze(0)  # Shape now is (1, C, H, W)

    # Use interpolate to resize
    resized_image_tensor = F.interpolate(image, size=target_size).clamp(min=0.0, max=1.0)
    
    return resized_image_tensor


def box_crop_2(box_min_size=0.05, box_max_size=0.99):
    """
    Apply a random crop to an image tensor within specified size bounds.

    Args:
        box_min_size (float, optional): Minimum size of the crop box as a fraction of the image dimensions. Defaults to 0.05.
        box_max_size (float, optional): Maximum size of the crop box as a fraction of the image dimensions. Defaults to 0.99.

    Returns:
        function: A function that applies the random crop to an image tensor.
    """
    def inner(image_tensor):
        """
        Perform a random crop on the given image tensor.

        Args:
            image_tensor (torch.Tensor): Tensor of image(s) to be cropped. Shape should be (B, C, W, H) where
                                         B is batch size, C is the number of channels, W is width, and H is height.

        Returns:
            torch.Tensor: The cropped image tensor.
        """
        image = image_tensor
        batch_size, num_channels, image_width, image_height = image.shape

        # Sample box size uniformly in [min_size, max_size]
        box_width_fraction = torch.rand(1) * (box_max_size - box_min_size) + box_min_size
        box_height_fraction = box_width_fraction

        # Sample top-left corner x0, y0 uniformly from 'in bounds' regions
        max_x0_fraction = 1 - box_width_fraction
        max_y0_fraction = 1 - box_height_fraction
        x0_fraction = torch.rand(1) * max_x0_fraction
        y0_fraction = torch.rand(1) * max_y0_fraction

        # Calculate pixel coordinates for the crop
        x_start = int(x0_fraction * image_width)
        x_end = int((x0_fraction + box_width_fraction) * image_width)
        y_start = int(y0_fraction * image_height)
        y_end = int((y0_fraction + box_height_fraction) * image_height)

        # Crop image
        cropped_image = image[:, :, x_start:x_end, y_start:y_end]

        return cropped_image

    return inner


def uniform_gaussian_noise(noise_std=0.02):
    """
    Add uniform and Gaussian noise to an image tensor.

    Args:
        noise_std (float, optional): Standard deviation of the noise to be added. Defaults to 0.02.

    Returns:
        function: A function that adds the specified noise to an image tensor.
    """
    def inner(image_tensor):
        """
        Add uniform and Gaussian noise to the given image tensor.

        Args:
            image_tensor (torch.Tensor): Tensor of image(s) to which noise will be added. Shape should be (B, C, H, W) where
                                         B is batch size, C is the number of channels, H is height, and W is width.

        Returns:
            torch.Tensor: The image tensor with added noise.
        """
        batch_size = image_tensor.shape[0]
        device = image_tensor.device

        # Generate Gaussian noise
        gaussian_noise = torch.normal(mean=0.0, std=noise_std, size=image_tensor[0].shape).to(device)
        gaussian_noise = gaussian_noise.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # Generate uniform noise
        uniform_noise = (torch.rand_like(image_tensor[0], dtype=torch.float32) - 0.5) * noise_std
        uniform_noise = uniform_noise.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # Add noise to the image tensor
        noisy_image_tensor = image_tensor + gaussian_noise + uniform_noise

        return noisy_image_tensor
    
    return inner


def box_crop(
    box_min_size=0.85,
    box_max_size=0.95,
    box_loc_std=0.05,
):
    """returns a function that will take an image and randomly crop it into a batch.

    Args:
        box_min_size (float, optional): minimum box size (as fraction of canevas size). Defaults to 0.10.
        box_max_size (float, optional): minimum box size (as fraction of canevas size). Defaults to 0.35.
        box_loc_std (float, optional): std of box positions (sampled normally around image center). Defaults to 0.2.
        noise_std (float, optional): std of the noise added to the image. Defaults to 0.05.
        model_input_size (tuple, optional): once the crop is made it is interpolated to this size

    Returns:
        (callable): a function that takes an image tensor of shape (b,c,h,w)
                    and returns a batch tensor of shape (b+nb_crops-1,c,h,w).
                    This function operates on the last image in the batch,
                    replacing it with cropped resized versions.
    """

    def inner(image_t):
        """
        Args: 
            x (model input tensor): 
            batch_idx
        Returns:
            batch tensor of shape (b,c,h,w)
        """


        x = image_t
        im_b, im_c, im_w, im_h = x.shape

        delta_x = torch.rand(1) * (box_max_size - box_min_size) + box_min_size
        delta_y = delta_x

        x0 = torch.clamp(
            torch.randn(1) * box_loc_std + 0.5, delta_x / 2.0, 1 - delta_x / 2.0
        )
        y0 = torch.clamp(
            torch.randn(1) * box_loc_std + 0.5, delta_y / 2.0, 1 - delta_y / 2.0
        )

        x = x[
                :, 
                :, 
                int((x0 - delta_x / 2.0) * im_w): int((x0 + delta_x / 2.0) * im_w), 
                int((y0 - delta_y / 2.0) * im_h): int((y0 + delta_y / 2.0) * im_h)
             ]

        return x

    return inner


standard_box_transforms = [
                           box_crop_2(),
                           uniform_gaussian_noise()
                          ]


def resize(target_size=default_model_input_size):
    """
    Resize an image tensor to the specified size.

    Args:
        target_size (tuple, optional): The target size for resizing the image tensor. Defaults to default_model_input_size.

    Returns:
        function: A function that resizes the image tensor to the specified size.
    """
    def inner(image_tensor):
        """
        Resize the given image tensor to the specified size.

        Args:
            image_tensor (torch.Tensor): Tensor of image(s) to be resized. Shape should be (B, C, H, W) where
                                         B is batch size, C is the number of channels, H is height, and W is width.

        Returns:
            torch.Tensor: The resized image tensor.
        """
        resized_image_tensor = F.interpolate(
            image_tensor,
            size=target_size,
            mode="bilinear",
            align_corners=True,
            antialias=True
        )
        return resized_image_tensor

    return inner


def range_normalize(normalization_range=default_model_input_range):
    """
    Normalize an image tensor to a specified range.

    Args:
        normalization_range (tuple, optional): The target range for normalization (min, max). Defaults to default_model_input_range.

    Returns:
        function: A function that normalizes the image tensor to the specified range.
    """
    def inner(image_tensor):
        """
        Normalize the given image tensor to the specified range.

        Args:
            image_tensor (torch.Tensor): Tensor of image(s) to be normalized. Shape should be (B, C, H, W) where
                                         B is batch size, C is the number of channels, H is height, and W is width.

        Returns:
            torch.Tensor: The normalized image tensor.
        """
        return image_tensor * (normalization_range[1] - normalization_range[0]) + normalization_range[0]
    
    return inner



def rebatch_transforms(tensor, num_transforms=1):
    """
    Add an extra dimension for transform batches.

    Args:
        tensor (torch.Tensor): The input tensor to be reshaped. Shape should be (N, C, H, W) where
                               N is batch size, C is the number of channels, H is height, and W is width.
        num_transforms (int, optional): The number of transform batches. Defaults to 1.

    Returns:
        torch.Tensor: The reshaped tensor with an added dimension for transform batches.
    """
    original_shape = list(tensor.shape)
    new_shape = [num_transforms, original_shape[0] // num_transforms] + original_shape[1:]
    reshaped_tensor = tensor.view(*new_shape)
    return reshaped_tensor


def compose(transformations, num_transforms=1):
    """
    Compose multiple transformations and apply them to an input tensor.

    Args:
        transformations (list): A list of transformation functions to be applied sequentially.
        num_transforms (int, optional): The number of times to apply the composed transformations. Defaults to 1.

    Returns:
        function: A function that applies the composed transformations to an input tensor.
    """
    def inner(input_tensor):
        """
        Apply the composed transformations to the input tensor.

        Args:
            input_tensor (torch.Tensor): The input tensor to which the transformations will be applied. Shape can vary.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        def apply_transformations(tensor):
            """
            Apply each transformation in the list to the tensor sequentially.

            Args:
                tensor (torch.Tensor): The tensor to be transformed.

            Returns:
                torch.Tensor: The transformed tensor.
            """
            for transformation in transformations:
                tensor = transformation(tensor)
            return tensor

        # Apply the transformations multiple times if specified
        if num_transforms > 1:
            transformed_tensors = [apply_transformations(input_tensor) for _ in range(num_transforms)]
            return torch.cat(transformed_tensors, dim=0)
        else:
            return apply_transformations(input_tensor)

    return inner