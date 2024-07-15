
# This part of the code is copied from the Lucent implementation on GitHub.
# They have complete copyrights over this part of the code.
# I have just used this for MaCO optimization. Only L2 function is written by me.

from __future__ import absolute_import, division, print_function

from utils import _make_arg_str, _extract_act_pos,batch_transform_handler

import numpy as np
import torch
import torch.nn.functional as F
from decorator import decorator


class Objective():

    def __init__(self, objective_func, name="", description=""):
        self.objective_func = objective_func
        self.name = name
        self.description = description

    def __call__(self, model):
        return self.objective_func(model)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            objective_func = lambda model: other + self(model)
            name = self.name
            description = self.description
        else:
            objective_func = lambda model: self(model) + other(model)
            name = ", ".join([self.name, other.name])
            description = "Sum(" + " +\n".join([self.description, other.description]) + ")"
        return Objective(objective_func, name=name, description=description)

    @staticmethod
    def sum(objs):
        objective_func = lambda T: sum([obj(T) for obj in objs])
        descriptions = [obj.description for obj in objs]
        description = "Sum(" + " +\n".join(descriptions) + ")"
        names = [obj.name for obj in objs]
        name = ", ".join(names)
        return Objective(objective_func, name=name, description=description)

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + (-1 * other)


    def __radd__(self, other):
        return self.__add__(other)

def wrap_objective():
    @decorator
    def inner(func, *args, **kwds):
        objective_func = func(*args, **kwds)
        objective_name = func.__name__
        args_str = " [" + ", ".join([_make_arg_str(arg) for arg in args]) + "]"
        description = objective_name.title() + args_str
        return Objective(objective_func, objective_name, description)
    return inner


def handle_batch(batch=None):
    return lambda f: lambda model: f(batch_transform_handler(model, batch_index=batch))


@wrap_objective()
def neuron(layer, unit, h=None, w=None, batch=None):
    
    @handle_batch(batch)
    def inner(model):
        layer_t = model(layer)
        layer_t = _extract_act_pos(layer_t, h, w)
        if isinstance(unit,int):
            return -layer_t[: ,:, unit].mean()
        else:
            return -dot_product(layer_t, unit.to(layer_t.device)).mean()

    return inner


@wrap_objective()
def channel(layer, unit, batch=None):
    """Visualize a single channel"""
    @handle_batch(batch)
    def inner(model):
        if isinstance(unit,int):
            return -model(layer)[: ,:, unit].mean()
        else:
            return -dot_product(model(layer), unit.to(model(layer).device)).mean()
    return inner


@wrap_objective()
def layer(layer_name, batch=None):
    """Visualize an entire layer."""
    @handle_batch(batch)
    def inner(model):
        layer_t = model(layer_name)
        return -layer_t.mean()

    return inner

def as_objective(obj):
    if isinstance(obj, Objective):
        return obj
    if callable(obj):
        return obj
    if isinstance(obj, str):
        parts = obj.split(":")
        if len(parts) == 2:
            layer_name, chn = parts[0].strip(), int(parts[1])
            return channel(layer_name, chn)
        elif len(parts) == 1:
            layer_name = parts[0].strip()
            return layer(layer_name)



@wrap_objective()
def l2_compare(layer, batch=0, comp_batch=1, p=2, pos=None):
    """
    Computes the L2 distance (or other p-norm distance) between two batches of 
    activations from a specified layer in a neural network.

    Args:
        layer (str): The name of the layer from which to extract activations.
        batch (int, optional): The index of the first batch to compare. Default is 0.
        comp_batch (int, optional): The index of the second batch to compare. Default is 1.
        p (int, optional): The norm degree (e.g., 2 for L2 norm). Default is 2.
        pos (tuple or str, optional): The position in the activation map to consider. 
                                      If 'middle', selects the center position. Default is None.

    Returns:
        callable: A function that takes a tensor T as input and computes the negative 
                  mean distance between the specified batches of activations.
    """
    def inner(T, pos=pos):
        x = T(layer)  # model output, with additional transform dimension
        x1 = x[:, batch]
        x2 = x[:, comp_batch]
        distances = torch.norm(x1 - x2, dim=1, p=p)

        if pos is not None:
            if pos == 'middle':
                pos = [distances.shape[1] // 2, distances.shape[2] // 2]
            distances = distances[:, pos[0], pos[1]]
        return -distances.mean()
    
    return inner





