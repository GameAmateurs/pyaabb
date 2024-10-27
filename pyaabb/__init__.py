"""Axis-aligned bounding box libarary."""
import numpy as np


def collisions(bboxes: np.array):
    """Identify all colliding bounding boxes in bboxes.

    Two bounding boxes are colliding if they are in contact.

    Arguments:
        bboxes: n x 2 x 2 array where n is the number of bounding boxes,
           [[[x1, y1], [x2, y2]]] where x1, y1 is the bottom left corner,
           x2 y2 the top right

    Returns:
        m by 2 array of collisions, where the elements are the indices of the
        colliding bboxes
    """
    # inefficient: can be improved to avoid comparisons
    #   between self and inverse collisions
    set1 = bboxes[:, np.newaxis]
    set2 = bboxes[np.newaxis]
    colliding = _identify_overlapping(set1, set2)
    colliding[np.tril_indices(colliding.shape[0])] = False
    colls = np.transpose(np.nonzero(colliding))
    return colls


def _identify_overlapping(set1, set2):
    # corners of bounding boxes
    lowerleft1 = set1[:, :, 0]
    lowerleft2 = set2[:, :, 0]
    upperright1 = set1[:, :, 1]
    upperright2 = set2[:, :, 1]

    overlap1 = (lowerleft1 < upperright2).all(axis=-1)
    overlap2 = (upperright1 > lowerleft2).all(axis=-1)

    return overlap1 & overlap2


X1, Y1, X2, Y2 = (0, 0), (0, 1), (1, 0), (1, 1)


def slide(box1, box2, vx, vy):
    """Resolve a collision between box1 and box2 with a 'slide' mechanic.

    A slide mechanic is one where the velocity perpendicular to the surfaces
    collision is set to 0, and the velocity parallel to the surface is not.

    Arguments:
        box1: 2 x 2 array [[x1, y1], [x2, y2]] of the first box
        box2: same for the second (static) box
        vx, vy are the velocity of the first box

    Returns:
        box1: position of box1 after collision is resolved
        vx, vy: velocity of box1 after collision is resolved
    """
    box1 = np.array(box1)
    box2 = np.array(box2)

    if (vx, vy) == (0, 0):
        return _pop_out_minimum_direction(box1, box2)

    ox, oy = _find_overlap_in_direction_of_movement(box1, box2, vx, vy)

    time_since_x_intersect = _find_intersection_time(ox, vx)
    time_since_y_intersect = _find_intersection_time(oy, vy)

    if time_since_x_intersect < time_since_y_intersect:
        return box1 + np.array([[0, oy]]), vx, 0
    return box1 + np.array([[ox, 0]]), 0, vy


HUGE = 1e90


def _find_intersection_time(overlap, velocity):
    if overlap == 0:
        intersect_time = 0
    elif velocity == 0:
        intersect_time = -HUGE
    else:
        intersect_time = overlap / velocity
    return intersect_time


def _find_overlap_in_direction_of_movement(box1, box2, vx, vy):
    if vx > 0:
        ox = box2[X1] - box1[X2]
    else:
        ox = box2[X2] - box1[X1]
    if vy > 0:
        oy = box2[Y1] - box1[Y2]
    else:
        oy = box2[Y2] - box1[Y1]
    return ox, oy


def _pop_out_minimum_direction(box1, box2):
    """Move box1 minimum necessary to not overlap box2."""
    ox1, ox2 = box2[X1] - box1[X2], box2[X2] - box1[X1]

    if np.abs(ox1) < np.abs(ox2):
        ox = ox1
    else:
        ox = ox2

    oy1, oy2 = box2[Y1] - box1[Y2], box2[Y2] - box1[Y1]
    if np.abs(oy1) < np.abs(oy2):
        oy = oy1
    else:
        oy = oy2

    if np.abs(ox) < np.abs(oy):
        return box1 + np.array([[ox, 0]]), 0, 0
    return box1 + np.array([[0, oy]]), 0, 0
