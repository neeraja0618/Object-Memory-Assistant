import math
import numpy as np


def displacement_warning(matrix,width,height):
    points=np.array([[0,0],[width,0],[width,height],[0,height]],dtype=np.float32)
    homogeneous=np.column_stack([points,np.ones(4)])@np.asarray(matrix).T
    denominator=homogeneous[:,2:3]
    if np.any(np.abs(denominator)<1e-8):
        return False
    moved=homogeneous[:,:2]/denominator
    shift=np.median(np.linalg.norm((moved-points)/[width,height],axis=1))
    edge=moved[1]-moved[0]
    scale=np.linalg.norm(edge)/width
    angle=abs(math.degrees(math.atan2(edge[1],edge[0])))
    return bool(shift>.055 or abs(scale-1)>.12 or angle>8)
