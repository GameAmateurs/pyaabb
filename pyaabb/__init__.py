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

X1, Y1, X2, Y2 = (0, 0), (0, 1), (1, 0), (1, 1)

# TODO: make this vectorized, allowing many slides
def slide(box1, box2, vx, vy):
    """Resolve a collision between box1 and box2 with a 'slide' mechanic.
    
    A slide mechanic is one where the velocity perpendicular to the surfaces collision is evened out,
    but the velocity parallel to the surface is not.
    
    Arguments:
        box1: 2 x 2 array [[x1, y1], [x2, y2]] of the first (moving, sliding) box
        box2: same for the second (static) box
        vx, vy are the velocity of the first box
        
    Returns:
        box1: position of box1 after collision is resolved
        vx, vy: velocity of box1 after collision is resolved
    """
    box1 = np.array(box1)
    box2 = np.array(box2)
    # TODO: test the negative cases
    # TODO: test and address the case where penetration is smaller in the colliding axis
    # TODO: ensure good test coverage: need to test aox < aoy as
    if (vx, vy) == (0, 0):
        ox = min(box2[X1] - box1[X2], box2[X2] - box1[X1])
        oy = min(box2[Y1] - box1[Y2], box2[Y2] - box1[Y1])
        
        if np.abs(ox) < np.abs(oy):
            return box1 + np.array([[ox, 0]]), 0, 0
        return  box1 + np.array([[0, oy]]), 0, 0
    import pdb; pdb.set_trace()
    if vx > 0:
        ox = box2[X1] - box1[X2]
    else:
        ox = box2[X2] - box1[X1]
    if vy > 0 :
        oy = box2[Y1] - box1[Y2]
    else:
        oy = box2[Y2] - box1[Y1]
        
    if ox == 0:
        xintersect = 0
    elif vx == 0:
        xintersect = -1e90
    else:
        xintersect = ox / vx
        
    if oy == 0:
        yintersect = 0
    elif vy == 0:
        yintersect = -1e90
    else:
        yintersect = oy / vy

    if xintersect < yintersect:
        # xaxis crossed before yaxis
        # collision occurs on vertical boundary
        return box1 + np.array([[0, oy]]), vx, 0
    # collision occurs on horizontal boundary
    return box1 + np.array([[ox, 0]]), 0, vy

# for each object 
# move fwd and find collisions
# see how far to reverse for each coll
# pick the max reverse one and yield it
#   ask how to resolve
#   