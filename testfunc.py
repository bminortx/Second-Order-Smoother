# Purpose: to convolute an image and resmooth it out using
# a second-order smoothing term
# This paper was a great help:
# http://bit.ly/1uMjmsN

import numpy as np
import scipy.linalg
import cv2



# Elegantly programmed norm from http://bit.ly/1xuPoqn
def huberLoss(eig_x, eig_y):
    c = 2;
    dy = 1;
    eigs = np.linalg.norm((eig_x[:], eig_y[:]), 2, axis=0);
    y_fit = np.ones((1, eig_x.shape[0]))
    print eigs
    print y_fit
    t = abs((eigs - y_fit) / dy);
    print t
    flag = t > c;
    # np.sum((~flag) * (0.5 * t ** 2) - (flag) * c * (0.5 * c - t), -1)
    return ((~flag) * (0.5 * t ** 2) - (flag) * c * (0.5 * c - t));


# Main
if __name__ == '__main__':
    # eigs_x = np.array([0, 3, 4]);
    # eigs_y = np.array([2, 5, 2]);
    # A = huberLoss(eigs_x, eigs_y);
    rows = 30;
    T_one = np.vstack([[ 0,  0],
                       [ 0,  1],
                       [ 0,  0],
                       np.zeros(rows-3, 2),
                       [ 1,  0],
                       [ 0,  0],
                       [-1,  0],
                       np.zeros(rows-3, 2),
                       [ 0,  0],
                       [ 0, -1],
                       [ 0,  0]])
    print T_one

