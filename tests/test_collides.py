import pyaabb
import numpy as np


def test_finds_overlapping_boxes():
    
    boxes = np.array([
        [[0, 0], [1, 1]], # collides with 1 and 2
        [[0.5, 0.5,], [1.5, 1.5]], # collides with 0 and 2
        [[1, 0,], [2, 1]], # collides with 1
        [[3, 3], [4, 4]]] # collides with none
    )
    
    colls = pyaabb.collisions(boxes)
    assert np.allclose(
        colls, [[0, 1], [1, 2]])

