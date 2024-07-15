from __future__ import absolute_import, division, print_function

from collections import OrderedDict
import torch
from transformations import rebatch_transforms

TORCH_VERSION = torch.__version__

class HookModel:
    """
    A class to hook into the layers of a model to capture intermediate features.

    Args:
        model (torch.nn.Module): The model to hook into.
        image_function (callable, optional): A function that provides the input image. Defaults to None.
        transform_function (callable, optional): A function to apply transformations. Defaults to None.
        num_transforms (int, optional): Number of transformations to apply. Defaults to None.
        layers (list, optional): List of layer names to hook into. Defaults to None.

    Returns:
        HookModel: An instance of the HookModel class.
    """
    def __init__(self, model, image_function=None, transform_function=None, num_transforms=None, layers=None):
        self.image_function = image_function
        self.num_transforms = num_transforms
        self.layers = layers
        self.model = model
        self.transform_function = transform_function
        self.features = OrderedDict()
        
        # Recursive hooking function
        def hook_layers(module, prefix=[], layers=self.layers):
            if hasattr(module, "_modules"):
                for name, layer in module._modules.items():
                    if layer is None:
                        continue

                    layer._forward_hooks.clear()
                    layer._forward_pre_hooks.clear()
                    layer._backward_hooks.clear()
                    
                    full_layer_name = "_".join(prefix + [name])
                    if (layers is None) or (full_layer_name in layers):
                        self.features[full_layer_name] = ModuleHook(layer)

                    hook_layers(layer, prefix=prefix + [name])

        hook_layers(model)

    def hook_function(self, layer_name, num_transforms=None):
        """
        Retrieve features from the specified layer.

        Args:
            layer_name (str): Name of the layer to retrieve features from.
            num_transforms (int, optional): Number of transformations to apply. Defaults to None.

        Returns:
            torch.Tensor: The features from the specified layer.
        """
        if layer_name == "input":
            output = self.transform_function(self.image_function())
        elif layer_name == "labels":
            output = list(self.features.values())[-1].features
        else:
            assert layer_name in self.features, f"Invalid layer {layer_name}. Retrieve the list of layers with `lucent.modelzoo.util.get_model_layers(model)`."
            output = self.features[layer_name].features

        if num_transforms is not None:
            return rebatch_transforms(output, num_transforms=num_transforms)
        elif self.num_transforms:
            return rebatch_transforms(output, num_transforms=self.num_transforms)
        else:
            return output

    def __exit__(self, *args):
        for feature in self.features.values():
            feature.close()

    def __enter__(self, *args):
        return self


class ModuleHook:
    """
    A class to hook into a specific layer of a model.

    Args:
        module (torch.nn.Module): The layer to hook into.

    Returns:
        ModuleHook: An instance of the ModuleHook class.
    """
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = None

    def hook_fn(self, module, input, output):
        """
        The hook function to capture the output of the layer.

        Args:
            module (torch.nn.Module): The layer being hooked.
            input (torch.Tensor): The input to the layer.
            output (torch.Tensor): The output of the layer.
        """
        self.module = module
        self.features = output

    def close(self):
        """
        Remove the hook from the layer.
        """
        self.hook.remove()