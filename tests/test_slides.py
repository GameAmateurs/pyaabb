import pyaabb
from pyaabb import X1, Y1, X2, Y2
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
    

def plot_situation(box1, box2, vx, vy):
       
    def box2rect(box):
        return Rectangle(
            (box[X1], box[Y1]),
            box[X2]-box[X1], 
            box[Y2] - box[Y1])
    
    
    f, a = plt.subplots(figsize=(5,5))
    a.add_collection(
        PatchCollection([box2rect(box1)], facecolor='red')
    )
    a.add_collection(
        PatchCollection([box2rect(box2)], facecolor='green')
    )
    a.add_collection(
        PatchCollection([box2rect(box1 - np.array([[vx, vy]]))], facecolor='red', alpha=0.2)
    )
    plt.arrow(*(box1[0] - np.array([vx, vy])), vx, vy)
    
    a.set_aspect('equal', adjustable='box')
    return f, a


def test_one_vel_zero():
    box1 = np.array([[0, 1], [1, 2]])
    box2 = np.array([[0, 0], [1, 1]])
    vx, vy = [0, -1]
    
    box1, vx, vy = pyaabb.slide(box1, box2, vx, vy)
    assert np.allclose(box1, np.array([[0, 1], [1, 2]]))
    assert vx == 0
    assert vy == 0

    
def test_vel_0():
    box1 = np.array([[0, 0], [1., 1.]])
    box2 = np.array([[0.6, 0.5], [1.6, 1.5]])

    plot_situation(box1, box2, 0, 0)
    plt.savefig(Path(__file__).parent / "test_slide_vel0.png")
    
    newbox1, newvx, newvy = pyaabb.slide(box1, box2, 0, 0)
    plot_situation(newbox1, box2, newvx, newvy)
    plt.savefig(Path(__file__).parent / "test_slide_vel0_after.png")
    
    assert np.allclose(newbox1, [[-0.4, 0], [0.6, 1]])
    assert newvx == 0
    assert newvy == 0

    newbox2, newvx, newvy = pyaabb.slide(box2, box1, 0, 0)
    assert np.allclose(newbox2, [[1, 0.5], [2, 1.5]])
    
    
def test_slide_in_y():
    box1 = np.array([[0., 0.], [1., 1.]])
    v = np.array([1, 0.25])
    box2 = np.array([[0.5, 1.1], [1.5, 2.6]])
    box1 += v

    plot_situation(box1, box2, *v)
    plt.savefig(Path(__file__).parent / "test_slide_y.png")
    
    newbox1, newvx, newvy = pyaabb.slide(box1, box2, *v)
    plot_situation(newbox1, box2, newvx, newvy)
    plt.savefig(Path(__file__).parent / "test_slide_y_after.png")
    
    assert np.allclose(newbox1, [[1, 0.1], [2, 1.1]])
    assert np.allclose([newvx, newvy], [1, 0])

 
def test_slide_in_x():
    
    box1 = np.array([[0., 0.], [1., 1.]])
    v = np.array([1, 0.25])
    box2 = np.array([[1.5, 0.5], [2.5, 1.5]])
    box1 += v

    plot_situation(box1, box2, *v)
    plt.savefig(Path(__file__).parent / "test_slide_x.png")
    
    newbox1, newvx, newvy = pyaabb.slide(box1, box2, *v)
    plot_situation(newbox1, box2, newvx, newvy)
    plt.savefig(Path(__file__).parent / "test_slide_x_after.png")
    
    assert np.allclose(newbox1, [[0.5, 0.25], [1.5, 1.25]])
    assert np.allclose([newvx, newvy], [0, 0.25])

def test_slide_neg_x():

    box1 = np.array([[1.5, 0.5], [2.5, 1.5]])
    box2 = np.array([[0., 0.], [1., 1.]])
    v = -np.array([1, 0.25])
    box1 += v

    plot_situation(box1, box2, *v)
    plt.savefig(Path(__file__).parent / "test_slide_negx.png")
    
    newbox1, newvx, newvy = pyaabb.slide(box1, box2, *v)
    plot_situation(newbox1, box2, newvx, newvy)
    plt.savefig(Path(__file__).parent / "test_slide_negx_after.png")
    
    assert np.allclose(newbox1, [[1., 0.25], [2, 1.25]])
    assert np.allclose([newvx, newvy], [0, -0.25])

    

# TODO: test velocity 0