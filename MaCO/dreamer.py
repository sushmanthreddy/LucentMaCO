from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import warnings
from MaCO.transformations import resize, range_normalize, compose
from preconditioning import FourierPhase
from image import plot, plot_alpha
from hooker import HookModel
from transformations import standard_box_transforms
from objective import as_objective,l2_compare


default_model_input_size = (224,224)
default_img_size = (512,512) 
default_model_input_range = (-2,2) 

def render(
    model,
    objective,
    parameterizer=None,
    transformations=None,
    optimizer_function=None,
    output_thresholds=range(0, 200, 2),
    inline_thresholds=range(0, 200, 10),
    transparency_percentile=20,
    num_transformations=16,
    initial_image=None,
    verbose=False,
    preprocess=True,
    show_progress=True,
    image_transform_objective=None,
    regularize_for_image_transform=False,
    model_input_size=None,
    model_input_range=None,
    image_size=None,
    hook_layers=None,
    accentuation_reg_layer=None,
    accentuation_reg_alpha=None,
    activation_cutoff=None,
    simple_transformations=None,
    pgd_alpha=None
):
    """
    Render images by optimizing the given objective function.

    Args:
        model (torch.nn.Module): The model to be used for rendering.
        objective (callable): The objective function to be optimized.
        parameterizer (callable, optional): The parameterizer for image generation. Defaults to None.
        transformations (list, optional): List of transformations to be applied. Defaults to None.
        optimizer_function (callable, optional): The optimizer function. Defaults to None.
        output_thresholds (range, optional): Thresholds for saving output images. Defaults to range(0, 200, 2).
        inline_thresholds (range, optional): Thresholds for inline plotting. Defaults to range(0, 200, 10).
        transparency_percentile (int, optional): Percentile for transparency. Defaults to 20.
        num_transformations (int, optional): Number of transformations to apply. Defaults to 16.
        initial_image (torch.Tensor or str, optional): Initial image for parameterization. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        preprocess (bool, optional): Whether to apply preprocessing. Defaults to True.
        show_progress (bool, optional): Whether to show progress bar. Defaults to True.
        image_transform_objective (callable, optional): Objective for image transformation. Defaults to None.
        regularize_for_image_transform (bool, optional): Whether to regularize for image transformation. Defaults to False.
        model_input_size (tuple, optional): Input size for the model. Defaults to None.
        model_input_range (tuple, optional): Input range for the model. Defaults to None.
        image_size (tuple, optional): Size of the generated images. Defaults to None.
        hook_layers (list, optional): Layers to hook for feature extraction. Defaults to None.
        accentuation_reg_layer (str, optional): Layer for accentuation regularization. Defaults to None.
        accentuation_reg_alpha (float, optional): Alpha value for accentuation regularization. Defaults to None.
        activation_cutoff (float, optional): Cutoff for activation. Defaults to None.
        simple_transformations (list, optional): Simple transformations to apply. Defaults to None.
        pgd_alpha (float, optional): Alpha value for PGD. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - imgs (list): List of generated images.
            - img_trs (list): List of image transformations.
            - losses (list): List of losses.
            - img_tr_losses (list): List of image transformation losses.
    """
    device = next(model.parameters()).device

    if len(inline_thresholds) >= 1:
        inline_thresholds = list(inline_thresholds)
        inline_thresholds.append(max(output_thresholds))

    if model_input_size is None:
        try:
            model_input_size = model.model_input_size
        except:
            print(f'warning: arg "model_input_size" not set, using default {default_model_input_size}')
            model_input_size = default_model_input_size

    if model_input_range is None:
        try:
            model_input_range = model.model_input_range
        except:
            print(f'warning: arg "model_input_range" not set, using default {default_model_input_range}')
            model_input_range = default_model_input_range

    if image_size is None:
        image_size = default_img_size

    if parameterizer is None:
        parameterizer = FourierPhase(device=device, image_size=image_size, init_image=initial_image)

    if initial_image is not None:
        print(f'Initializing parameterization with {initial_image}')

    if accentuation_reg_layer is not None:
        if accentuation_reg_alpha is None:
            print('Accentuation reg_layer arg set, but not reg_alpha, using reg_alpha=1')
            accentuation_reg_alpha = 1.0
        parameterizer.forward_init_img = True

    if accentuation_reg_alpha is not None and accentuation_reg_layer is None:
        print('arg reg_alpha set, but not reg_layer, you must set reg_layer! Applying no regularization.')

    params, img_f = parameterizer()
    for p in params:
        p.requires_grad_(True)

    for p in model.parameters():
        p.requires_grad = False

    if optimizer_function is None:
        if pgd_alpha is None:
            optimizer_function = lambda parameters: torch.optim.Adam(parameters, lr=0.05)
        else:
            optimizer_function = lambda parameters: torch.optim.SGD(parameters, lr=0.05)

    optimizer = optimizer_function(params)

    if transformations is None:
        try:
            print('Using parameterizer.standard_transforms')
            transformations = parameterizer.standard_transforms.copy()
        except:
            transformations = standard_box_transforms.copy()
    transformations = transformations.copy()

    if preprocess:
        transformations.append(resize(model_input_size))
        transformations.append(range_normalize(model_input_range))
    transform_function = compose(transformations, num_transforms=num_transformations)

    if simple_transformations is None:
        simple_transform_function = compose(
            [resize(model_input_size), range_normalize(model_input_range)],
            num_transforms=num_transformations
        )
    else:
        simple_transform_function = compose(simple_transformations, num_transforms=num_transformations)

    if pgd_alpha is not None:
        pgd_init_img = img_f().detach().clone().requires_grad_(False)[:1]

    with HookModel(model, img_f, transform_function, num_transforms=num_transformations, layers=hook_layers) as hooker:
        hook = hooker.hook_function

        img = img_f()

        imgs = []
        img_trs = []
        img_tr = torch.zeros(img.shape).float().cpu()
        losses = []
        img_tr_losses = []

        objective = as_objective(objective)

        if verbose or accentuation_reg_layer is not None:
            model(transform_function(img))
            print("Initial loss: {:.3f}".format(objective(hook)))

        if accentuation_reg_layer is not None:
            accentuation_reg_obj = l2_compare(accentuation_reg_layer)

            reg_loss = accentuation_reg_obj(hook)
            reg_loss.backward(retain_graph=True)
            reg_grad = sum(float(torch.sum(torch.abs(p.grad.detach().clone())).cpu()) for p in params)

            optimizer.zero_grad()
            for p in params:
                p.grad.zero_()

            obj_loss = objective(hook)
            obj_loss.backward(retain_graph=True)
            obj_grad = sum(float(torch.sum(torch.abs(p.grad.detach().clone())).cpu()) for p in params)

            optimizer.zero_grad()
            for p in params:
                p.grad.zero_()

            print(f'Objective/Regularization ratio: {obj_grad / reg_grad}')
            print('Setting regularization balance parameter to this ratio')
            accentuation_reg_balance = obj_grad / reg_grad

        if image_transform_objective is not None:
            image_transform_objective = objectives.as_objective(image_transform_objective)
        elif (accentuation_reg_layer is not None) and not regularize_for_image_transform:
            image_transform_objective = objective

        if accentuation_reg_layer is not None:
            objective_full = objective - accentuation_reg_alpha * accentuation_reg_balance * accentuation_reg_obj
        else:
            objective_full = objective

        for i in tqdm(range(0, max(output_thresholds) + 1), disable=not show_progress):
            optimizer.zero_grad()
            img = img_f()
            img.retain_grad()

            try:
                model(transform_function(img))
            except RuntimeError as ex:
                if i == 1:
                    warnings.warn(
                        "Some layers could not be computed because the size of the "
                        "image is not big enough. It is fine, as long as the non"
                        "computed layers are not used in the objective function"
                        f"(exception details: '{ex}')"
                    )

            if image_transform_objective is not None:
                img_tr_loss = image_transform_objective(hook)
                img_tr_loss.backward(retain_graph=True)

                img_tr = img_tr.detach().clone() + torch.abs(img.grad.detach().clone().cpu())
                img.grad.zero_()
                optimizer.zero_grad()
                for p in params:
                    p.grad.zero_()

            loss = objective_full(hook)
            loss.backward()

            if image_transform_objective is None:
                img_tr = img_tr.detach().clone() + torch.abs(img.grad.detach().clone().cpu())

            if i in output_thresholds:
                imgs.append(img.detach().clone().cpu().requires_grad_(False))
                img_trs.append(img_tr)
                losses.append(-float(loss.detach().clone().cpu()))
                if image_transform_objective is not None:
                    img_tr_losses.append(-float(img_tr_loss.detach().clone().cpu()))

            if i in inline_thresholds:
                if transparency_percentile is None or transparency_percentile in (0, 100):
                    plot(img)
                else:
                    plot_alpha(img, img_tr, percentile=transparency_percentile)

            if activation_cutoff is not None:
                model(simple_transform_function(img))
                if -objective(hook) > activation_cutoff:
                    if i not in output_thresholds:
                        imgs.append(img.detach().clone().cpu().requires_grad_(False))
                        img_trs.append(img_tr)
                        losses.append(-float(loss.detach().clone().cpu()))
                        if image_transform_objective is not None:
                            img_tr_losses.append(-float(img_tr_loss.detach().clone().cpu()))

                    if transparency_percentile is None or transparency_percentile in (0, 100):
                        plot(img)
                    else:
                        plot_alpha(img, img_tr, percentile=transparency_percentile)

                    return imgs, img_trs, losses, img_tr_losses

            optimizer.step()

            if pgd_alpha is not None:
                img = parameterizer.params_to_img().detach().clone().requires_grad_(False)[:1]
                pgd_d = img - pgd_init_img
                pgd_d_norm = torch.norm(pgd_d)
                if pgd_d_norm > pgd_alpha:
                    print('Projecting')
                    img = pgd_init_img + (pgd_alpha / pgd_d_norm) * pgd_d
                    parameterizer.img_to_params(img)
                    params, img_f = parameterizer()
                    for p in params:
                        p.requires_grad_(True)
                    optimizer = optimizer_function(params)

    return imgs, img_trs, losses, img_tr_losses

