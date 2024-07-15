
import torch

def _make_arg_str(arg):
    arg = str(arg)
    too_big = len(arg) > 15 or "\n" in arg
    return "..." if too_big else arg


def _extract_act_pos(acts, h=None, w=None):
    shape = acts.shape
    h = shape[3] // 2 if h is None else h
    w = shape[4] // 2 if w is None else w
    return acts[: ,:, :, h:h+1, w:w+1]


def batch_transform_handler(transform_func, batch_index=None):
    """
    Handle batching for a transformation function.

    Args:
        transform_func (callable): The transformation function to be applied.
        batch_index (int, optional): The specific batch index to handle. If None, handle all batches. Defaults to None.

    Returns:
        callable: A function that applies the transformation to the specified batch or all batches.
    """
    def apply_transform(name):
        """
        Apply the transformation to the specified batch or all batches.

        Args:
            name (str): The name of the transformation.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        transformed_tensor = transform_func(name)
        if isinstance(batch_index, int):
            return transformed_tensor[:, batch_index:batch_index + 1]
        else:
            return transformed_tensor
        
    return apply_transform


def linconv_reshape(x,ref):
    #reshapes x to shape of y where x is shape (c) or (b,c) and y is either shape (b,c) or (b,c,h,w)
    if x.dim() == 1: x = x.unsqueeze(0)
    while x.dim() < ref.dim():
        x = x.unsqueeze(dim=-1)
    return x



    
