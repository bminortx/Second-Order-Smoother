# Purpose: to convolute an image and resmooth it out using
# a second-order smoothing term
# This paper was a great help:
# http://bit.ly/1uMjmsN
# OpenCV makes it easy to do these kinds of operations:
# http://bit.ly/1vZAjSU

import numpy as np
import scipy.linalg
import cv2
import time


# Apply some gaussian blur to this biznitch
def blurImage(fn):
  ksize = (9, 9)
  sigma_x = 4
  sigma_y = 4
  blurimg = cv2.GaussianBlur(fn, ksize, sigma_x, sigma_y, cv2.BORDER_REPLICATE)
  cv2.imshow("bin", blurimg)
  cv2.imwrite("./blurimg.jpg", blurimg)
  # Just a placeholder
  cv2.imwrite("./rectimg.jpg", blurimg)
  return blurimg

# Elegantly programmed norm from http://bit.ly/1xuPoqn
# Do this element-wise through the full image
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
  # We don't want the sum, just the element-wise norm
  # np.sum((~flag) * (0.5 * t ** 2) - (flag) * c * (0.5 * c - t), -1)
  return ((~flag) * (0.5 * t ** 2) - (flag) * c * (0.5 * c - t));


# Main
if __name__ == '__main__':
  fn = cv2.imread('./tree.jpg', 0)
  blurimg = blurImage(fn)
  rows, cols = blurimg.shape

  #################
  # CALCULATE DERIVATIVES
  #################
  # Our tensors for second-order operations
  # First column is x operations, second is y.
  # A_one: The laplacian
  # http://bit.ly/12je4JY
  A_one = cv2.Laplacian(blurimg, cv2.CV_64F)

  # A_two: The negative laplacian? Something like that.
  # It's the summation of two convolutions
  # Horizontally
  kernel = [[0,  0,  0],
            [1, -2,  1],
            [0,  0,  0]];
  kernel = np.asanyarray(kernel);
  A_horz = cv2.filter2D(blurimg, -1, kernel);
  # Vertically
  kernel = [[0,  1,  0],
            [0, -2,  0],
            [0,  1,  0]];
  kernel = np.asanyarray(kernel);
  A_vert = cv2.filter2D(blurimg, -1, kernel);
  A_two = A_horz - A_vert;
  cv2.imwrite("A_two.png", A_two)

  # A_three: Partial derivatives
  # Convolute in horz, and use that result in convolution of vert
  kernel = [[0,  0,  0],
            [1,  0, -1],
            [0,  0,  0]];
  kernel = np.asanyarray(kernel);
  A_partial_horz = cv2.filter2D(blurimg, -1, kernel);
  kernel = [[0,  1,  0],
            [0,  0,  0],
            [0, -1,  0]];
  kernel = np.asanyarray(kernel);
  A_three = cv2.filter2D(A_partial_horz, -1, kernel);
  cv2.imwrite("A_three.png", A_partial)

  # Well that was stupid easy.

  #################
  # CALCULATE EIGENVALUES
  #################

  max_eigs = A_one + np.sqrt(np.square(A_two) + np.square(A_three));
  min_eigs = A_one - np.sqrt(np.square(A_two) + np.square(A_three));
  diff_eigs = max_eigs - min_eigs;

  # print "Found max and min eigenvalues"

  # # Calculate the difference in eigenvalues
  # diff_eigs_x = max_eigs_x - min_eigs_x;
  # diff_eigs_y = may_eigs_y - min_eigs_y;

  # print "Found eigenvalue differences"

  # # Find J_new
  # huber_max_eigs = huberLoss(max_eigs_x, max_eigs_y)
  # huber_min_eigs = huberLoss(min_eigs_x, min_eigs_y)
  # huber_diff_eigs = huberLoss(diff_eigs_x, diff_eigs_y)

  # print "Found Huber norm of eigen pairs"

  # J_new = np.sum(.5 * (huber_max_eigs + huber_min_eigs + huber_diff_eigs));

  # print "Found J_new, the reggularization term"

  # # Find grad_J_new
  # g_x = np.sqrt( pow(A_one_x, 2) + pow(A_two_x, 2));
  # g_y = np.sqrt( pow(A_one_y, 2) + pow(A_two_y, 2));

  # print "Found g"

  # M_one_x = np.diag((1 + np.sign(diff_eigs_x) / huber_max_eigs)
  #                   + (1 - np.sign(diff_eigs_x) / huber_max_eigs))
  # M_one_y = np.diag((1 + np.sign(diff_eigs_y) / huber_max_eigs)
  #                   + (1 - np.sign(diff_eigs_y) / huber_max_eigs))
  # M_two_x = np.diag((1 + np.sign(diff_eigs_x) / huber_max_eigs)
  #                   - (1 - np.sign(diff_eigs_x) / huber_max_eigs))
  # M_two_y = np.diag((1 + np.sign(diff_eigs_y) / huber_max_eigs)
  #                   - (1 - np.sign(diff_eigs_y) / huber_max_eigs))

  # print "Found M1 and M2"

  # Theta_x = np.diag( ((np.dot(np.sign(max_eigs_x),
  #                             (1 + np.sign(diff_eigs_x))) / g_x)
  #                     - (np.dot(np.sign(min_eigs_x),
  #                               (1 + np.sign(diff_eigs_x))) / g_x)))
  # Theta_y = np.diag( ((np.dot(np.sign(max_eigs_y),
  #                             (1 + np.sign(diff_eigs_y))) / g_y)
  #                     - (np.dot(np.sign(min_eigs_y),
  #                               (1 + np.sign(diff_eigs_y))) / g_y)))

  # print "Found Theta"

  # # Find grad_J, used in the final optimization
  # # gradJ = .5[T1' M1 T1 + T2' Theta T2 + T3' Theta T3] f + .5 [T1' M2 g]
  # grad_J_x = .5 * np.dot((np.dot(np.dot(T_one_x.transpose(), M_one_x),
  #                                T_one_x)
  #                         + np.dot(np.dot(T_two_x.transpose(), Theta_x),
  #                                  T_two_x)
  #                         + np.dot(np.dot(T_three_x.transpose(), Theta_x),
  #                                  T_three_x)),
  #                        f) + .5 * np.dot(np.dot(T_one_x.transpose(),
  #                                                M_one_x), g_x);
  # grad_J_y = .5 * np.dot((np.dot(np.dot(T_one_y.transpose(), M_one_y),
  #                                T_one_y)
  #                         + np.dot(np.dot(T_two_y.transpose(), Theta_y),
  #                                  T_two_y)
  #                         + np.dot(np.dot(T_three_y.transpose(), Theta_y),
  #                                  T_three_y)),
  #                        f) + .5 * np.dot(np.dot(T_one_y.transpose(),
  #                                                M_one_y), g_y);

  # print "Found the gradient!"

  # # Perform the optimization
  # print grad_J_x.shape[0]
  # print grad_J_y.shape[0]

  ###################
  # /Optimization iteration here
  ###################
