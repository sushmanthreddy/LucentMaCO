from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn.functional as F
from transformations import img_to_img_tensor, default_img_size, box_crop, uniform_gaussian_noise,standard_box_transforms
from einops import rearrange


imagenet_color_correlation = torch.tensor(
    [[0.56282854, 0.58447580, 0.58447580],
     [0.19482528, 0.00000000, -0.19482528],
     [0.04329450, -0.10823626, 0.06494176]]
).float()

imagenet_color_correlation_inv = torch.linalg.inv(imagenet_color_correlation)

def recorrelate_colors(image_tensor):
    """
    Recorrelate the colors of an image tensor using predefined color correlation matrices.

    Args:
        image_tensor (torch.Tensor): The input image tensor. Shape should be (B, C, H, W) where
                                     B is batch size, C is the number of channels, H is height, and W is width.

    Returns:
        torch.Tensor: The image tensor with recolored channels.
    """
    image_tensor = rearrange(image_tensor, "b c h w -> b h w c")
    batch_size, height, width, num_channels = image_tensor.shape
    image_tensor_flat = rearrange(image_tensor, 'b h w c -> b (h w) c')
    image_tensor_flat = torch.matmul(image_tensor_flat, imagenet_color_correlation.to(image_tensor.device))
    image_tensor = rearrange(image_tensor_flat, 'b (h w) c -> b h w c', c=num_channels, w=width, h=height)
    image_tensor = rearrange(image_tensor, "b h w c -> b c h w")
    return image_tensor


class FourierPhase:
    """
    A class to handle Fourier phase transformations on images with magnitude constraints.

    Args:
        init_image (str or PIL.Image or torch.Tensor, optional): Initial image to initialize the parameters.
        forward_init_image (bool, optional): Whether to forward the initial image. Defaults to False.
        image_size (tuple, optional): Target size for images. Defaults to default_img_size.
        device (torch.device, optional): Device to use for computations. Defaults to None.
        batch_size (int, optional): Batch size. Defaults to 1.
        desaturation (float, optional): Desaturation factor. Defaults to 1.0.
        seed (int, optional): Random seed for initialization. Defaults to None.
        normalize_image (bool, optional): Whether to normalize the image. Defaults to None.
        normalize_phase (bool, optional): Whether to normalize the phase. Defaults to None.
        magnitude_spectrum_border (bool, optional): Whether to use magnitude spectrum border. Defaults to False.
        use_magnitude_alpha (bool, optional): Whether to use magnitude alpha. Defaults to None.
        copy_batch (bool, optional): Whether to copy the batch. Defaults to False.
        magnitude_alpha_init (float, optional): Initial value for magnitude alpha. Defaults to 5.0.
        color_decorrelate (bool, optional): Whether to decorrelate colors. Defaults to True.
        correlation_file_path (str, optional): Path to the correlation file. Defaults to "/kaggle/input/mini-007/imagenet_decorrelated.npy".
        name (str, optional): Name of the instance. Defaults to 'fourier_phase'.

    Returns:
        FourierPhase: An instance of the FourierPhase class.
    """

    def __init__(self,
                 init_image=None,
                 forward_init_image=False,
                 image_size=default_img_size,
                 device=None,
                 batch_size=1,
                 desaturation=1.0,
                 seed=None,
                 normalize_image=None,
                 normalize_phase=None,
                 magnitude_spectrum_border=False,
                 use_magnitude_alpha=None,
                 copy_batch=False,
                 magnitude_alpha_init=5.0,
                 color_decorrelate=True,
                 correlation_file_path="/kaggle/input/mini-007/imagenet_decorrelated.npy",
                 name='fourier_phase'):
        
        self.name = name
        self.correlation_file_path = correlation_file_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.desaturation = desaturation
        self.forward_init_image = forward_init_image
        self.magnitude_spectrum_border = magnitude_spectrum_border
        self.color_decorrelate = color_decorrelate
        self.copy_batch = copy_batch

        if normalize_image is None:
            normalize_image = init_image is None
        if normalize_phase is None:
            normalize_phase = init_image is None
        if use_magnitude_alpha is None:
            use_magnitude_alpha = init_image is not None

        self.normalize_image = normalize_image
        self.normalize_phase = normalize_phase
        self.use_magnitude_alpha = use_magnitude_alpha
        self.magnitude_alpha_init = magnitude_alpha_init
        self.seed = seed

        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device

        if init_image is None:
            self.init_image = init_image
        else:
            self.init_image = img_to_img_tensor(init_image, size=self.image_size).to(self.device)

        # Initialize parameters
        if init_image is None:
            print('No image specified, initializing with random phase and magnitude from "imagenet_decorrelated.npy".')
            self.random_init()
            self.standard_transforms = [box_crop(box_min_size=0.05, box_max_size=0.5, box_loc_std=0.1),
                                        uniform_gaussian_noise()]
        else:
            self.img_to_params(init_image)
            self.standard_transforms = standard_box_transforms

    def random_init(self, img_size=None):
        """
        Initialize random phase and load magnitude from a file.

        Args:
            img_size (tuple, optional): Size of the image. Defaults to None.
        """
        if img_size is None:
            img_size = self.image_size

        buffer_shape = (img_size[0], 1 + (img_size[1] // 2))

        if self.seed is not None:
            np.random.seed(self.seed)
        
        phase = torch.tensor(np.random.uniform(low=-np.pi, high=np.pi, size=(self.batch_size, 3) + buffer_shape), dtype=torch.float32).to(self.device)

        if self.copy_batch:
            single_phase = np.random.uniform(low=-np.pi, high=np.pi, size=(3,) + buffer_shape)
            phase = torch.tensor(single_phase, dtype=torch.float32).unsqueeze(0).repeat(self.batch_size, 1, 1, 1).to(self.device)
        magnitude = torch.stack([torch.tensor(np.load(self.correlation_file_path), dtype=torch.float32).to(self.device) for _ in range(self.batch_size)])

        magnitude = F.interpolate(magnitude, buffer_shape, mode="bilinear", align_corners=True, antialias=True) * ((magnitude.shape[-2] * magnitude.shape[-1]) / (buffer_shape[0] * buffer_shape[1]))
        
        self.magnitude = magnitude 
        self.params = [phase]

        if self.use_magnitude_alpha:
            mag_alpha = self.magnitude_alpha_init * torch.ones(magnitude.shape).to(self.device)
            self.params.append(mag_alpha)

    def img_to_params(self, img):
        """
        Convert an image in pixel space to polar coordinate frequency domain.

        Args:
            img (str or PIL.Image or torch.Tensor): The image in pixel space.

        Returns:
            list: List containing phase tensor and optionally magnitude alpha tensor.
        """
        img = img_to_img_tensor(img, size=self.image_size).to(self.device)
        img *= self.desaturation
        buffer = torch.fft.rfft2(img)
        magnitude = torch.abs(buffer)
        phase = torch.atan2(buffer.imag, buffer.real)

        if self.normalize_phase:
            phase = phase * (torch.std(phase) + 1e-4) + torch.mean(phase)

        phase = phase.repeat(self.batch_size, 1, 1, 1)
        magnitude = magnitude.repeat(self.batch_size, 1, 1, 1)

        self.magnitude = magnitude 
        self.params = [phase]

        if self.use_magnitude_alpha:
            mag_alpha = self.magnitude_alpha_init * torch.ones(magnitude.shape).to(self.device)
            self.params.append(mag_alpha)

        return self.params

    def params_to_img(self):
        """
        Convert the buffer in frequency domain to spatial domain.

        Returns:
            torch.Tensor: The image in pixel space.
        """
        phase = self.params[0]
        magnitude = self.magnitude
        if self.use_magnitude_alpha:
            magnitude = magnitude * torch.sigmoid(self.params[1])

        buffer = torch.complex(torch.cos(phase) * magnitude, torch.sin(phase) * magnitude)
        
        img = torch.fft.irfft2(buffer)
        img = img / self.desaturation

        if self.color_decorrelate:
            img = recorrelate_colors(img)
        if self.normalize_image:
            img = img - torch.mean(img, dim=(1, 2, 3), keepdim=True)
            img = img / (torch.std(img, dim=(1, 2, 3), keepdim=True) + 1e-4)

        img = torch.sigmoid(img)
        return img
    
    def __call__(self):
        """
        Return the parameters and an image generation function.

        Returns:
            tuple: A tuple containing the parameters and an image generation function.
        """
        if self.forward_init_image:
            img_func = lambda: torch.cat([self.params_to_img(), self.init_image], dim=0)
        else:
            img_func = self.params_to_img
        return self.params, img_func