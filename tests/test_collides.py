import pyaabb
import numpy as np


def test_finds_overlapping_boxes():
    
    boxes = np.array([
        [[0, 0], [1, 1]], # collides with 1
        [[0.5, 0.5,], [1.5, 1.5]], # collides with 0 and 2
        [[1, 0,], [2, 1]], # collides with 1
        [[3, 3], [4, 4]]] # collides with none
    )
    
    colls = pyaabb.collisions(boxes)
    assert np.allclose(
        colls, [[0, 1], [1, 2]])


def test_time_to_collisions():
    rel_v = np.array([[1, 0], [1, 0]])
    boxes = np.array([
        [[0.6, 0], [1.6, 1]],
        [[1.1, 0], [1.1, 2]],
        [[1.5, 0], [2, 1]]
    ])
    collision_times = pyaabb.time_to_collisions(
        boxes, 
        [[0, 1], [0, 2]], 
        rel_v
    )
    assert np.allclose(
        collision_times, [-0.5, -0.1]
    )
