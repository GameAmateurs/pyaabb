import numpy as np

def collisions(bboxes: np.array):
    """Identify all colliding bounding boxes in bboxes.
    
    Two bounding boxes are colliding if they overlap or their sides are in contact.
    
    Arguments:
        bboxes: n x 2 x 2 array where n is the number of bounding boxes,
           [[[x1, y1], [x2, y2]]] where x1, y1 is the bottom left corner,
           x2 y2 the top right
           
    Returns:
        m by 2 array of collisions, where the elements are the indices of the colliding
        bboxes
    """
    # inefficient: can be improved to avoid comparisons
    #   between self and inverse collisions
    set1 = bboxes[:, np.newaxis]
    set2 = bboxes[np.newaxis]
    colliding = (
        (set1[:, :, 0] <= set2[:, :, 1]).all(axis=-1)
        & (set1[:, :, 1] >= set2[:, :, 0]).all(axis=-1)
    )
    colliding[np.tril_indices(colliding.shape[0])] = False
    colls = np.transpose(np.nonzero(colliding))
    
    return colls