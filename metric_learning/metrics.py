def map_at_k_per_im(label, predictions, max_preds=5):
    """Computes the precision score of one image.

    Parameters
    ----------
    label : string
            The true label of the image
    predictions : list
            A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """    
    try:
        return 1 / (predictions[:max_preds].index(label) + 1)
    except ValueError:
        return 0.0

