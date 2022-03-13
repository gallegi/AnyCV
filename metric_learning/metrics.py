"""This module contains functions and classes for metric learning evaluation metric."""

from typing inport List

def map_at_k_per_im(label:str, predictions:List[str], max_preds:int=5):
    """Computes the precision score of one image.

    Args:
        label: the true label of the image
        predictions : a list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns:
        a double number of MAP at k score
    """    
    try:
        return 1 / (predictions[:max_preds].index(label) + 1)
    except ValueError:
        return 0.0

