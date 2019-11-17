# --------------------------------------------------------
# Edited by DEV KIM ---
# using numba v0.45 for acceleration
# --------------------------------------------------------

import numba
import numpy as np
from numba import njit

@njit(cache=True)
def anchors_cython(height, width, stride, base_anchors):
    """
    Parameters
    ----------
    height: height of plane
    width:  width of plane
    stride: stride ot the original image
    anchors_base: (A, 4) a base set of anchors
    Returns
    -------
    all_anchors: (height, width, A, 4) ndarray of anchors spreading over the plane
    """
    # using local variables in float32
    A = base_anchors.shape[0]
    all_anchors = np.zeros((height, width, A, 4), dtype=numba.float32)
    for iw in range(width):
        sw = iw * stride
        for ih in range(height):
            sh = ih * stride
            for k in range(A):
                all_anchors[ih, iw, k, 0] = base_anchors[k, 0] + sw
                all_anchors[ih, iw, k, 1] = base_anchors[k, 1] + sh
                all_anchors[ih, iw, k, 2] = base_anchors[k, 2] + sw
                all_anchors[ih, iw, k, 3] = base_anchors[k, 3] + sh
    return all_anchors