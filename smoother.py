# Purpose: to convolute an image and resmooth it out using
# a second-order smoothing term
# This paper was a great help:
# http://bit.ly/1uMjmsN
# OpenCV makes it easy to do these kinds of operations:
# http://bit.ly/1vZAjSU

import numpy as np
import scipy.linalg
import cv2


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


def createTMatrices(T_vector, size_of_img):
  print size_of_img
  T_final = np.zeros((size_of_img, 1));
  # Gotta do some weird things to turn matrix into array
  T_final[0:T_vector.shape[0], 0] = np.array(T_vector.T)[0];
  print T_final.shape
  return scipy.linalg.circulant(T_final)


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
  print blurimg.shape
  rows,cols = blurimg.shape
  print rows

  # Our tensors for second-order operations
  # First column is x operations, second is y.
  # T_one: The laplacian
  # http://bit.ly/12je4JY
  laplacian = cv2.Laplacian(blurimg, cv2.CV_64F)

  T_one = np.vstack([[ 0,  0],
                     [ 0,  1],
                     [ 0,  0],
                     np.zeros((rows-3, 2)),
                     [ 1,  0],
                     [ 0,  0],
                     [-1,  0],
                     np.zeros((rows-3, 2)),
                     [ 0,  0],
                     [ 0, -1],
                     [ 0,  0]])
  # T_two: The double derivative in one direction
  T_two = np.vstack([[  0,    0],
                     [  0,  .25],
                     [  0,    0],
                     np.zeros((rows-3, 2)),
                     [ .25,   0],
                     [-.5,  -.5],
                     [ .25,   0],
                     np.zeros((rows-3, 2)),
                     [  0,    0],
                     [  0,  .25],
                     [  0,    0]])
  # T_three: The partial derivative in both directions
  # Notice the x and y vectors are the same.
  T_three = np.vstack([[ .5,  .5],
                       [  0,   0],
                       [-.5, -.5],
                       np.zeros((rows-3, 2)),
                       [  0,   0],
                       [  0,   0],
                       [  0,   0],
                       np.zeros((rows-3, 2)),
                       [-.5, -.5],
                       [  0,   0],
                       [ .5,  .5]])
  max_eigs_x = np.zeros((rows * cols, 1))
  max_eigs_y = np.zeros((rows * cols, 1))
  min_eigs_x = np.zeros((rows * cols, 1))
  min_eigs_y = np.zeros((rows * cols, 1))
  diff_eigs_x = np.zeros((rows * cols, 1))
  diff_eigs_y = np.zeros((rows * cols, 1))
  # f: Image data, rearranged into a single row
  f = np.reshape(blurimg, (rows * cols, 1));
  # Our bound for the optimization
  epsilon = .001;

  print "Straightened out image"

  ###################
  # Optimization iteration here
  ###################

  # Calculate max and min eigenvalues from above values
  # Do this for x and y
  T_one_x = createTMatrices(T_one[:, 0], rows * cols);
  T_one_y = createTMatrices(T_one[:, 1], rows * cols);
  T_two_x = createTMatrices(T_two[:, 0], rows * cols);
  T_two_y = createTMatrices(T_two[:, 1], rows * cols);
  T_three_x = createTMatrices(T_three[:, 0], rows * cols);
  T_three_y = createTMatrices(T_three[:, 1], rows * cols);

  print "Created circular matrices"

  # m = rows * cols
  # [m x m] * [m x 1] = [m x 1]
  A_one_x = np.dot(T_one_x, f);
  A_one_y = np.dot(T_one_y, f);
  A_two_x = np.dot(T_double_f_x, f);
  A_two_y = np.dot(T_double_f_y, f);
  A_three_x = np.dot(T_split_f_x, f);
  A_three_y = np.dot(T_split_f_y, f);
  max_eigs_x = A_one_x + np.sqrt( pow(A_two_x, 2)
                                      + pow(A_three_x, 2))
  min_eigs_x = A_one_x - np.sqrt( pow(A_two_x, 2)
                                      + pow(A_three_x, 2))
  max_eigs_y = A_one_y + np.sqrt( pow(A_two_y, 2)
                                      + pow(A_three_y, 2))
  min_eigs_y = A_one_y - np.sqrt( pow(A_two_y, 2)
                                      + pow(A_three_y, 2))

  print "Found max and min eigenvalues"

  # Calculate the difference in eigenvalues
  diff_eigs_x = max_eigs_x - min_eigs_x;
  diff_eigs_y = may_eigs_y - min_eigs_y;

  print "Found eigenvalue differences"

  # Find J_new
  huber_max_eigs = huberLoss(max_eigs_x, max_eigs_y)
  huber_min_eigs = huberLoss(min_eigs_x, min_eigs_y)
  huber_diff_eigs = huberLoss(diff_eigs_x, diff_eigs_y)

  print "Found Huber norm of eigen pairs"

  J_new = np.sum(.5 * (huber_max_eigs + huber_min_eigs + huber_diff_eigs));

  print "Found J_new, the reggularization term"

  # Find grad_J_new
  g_x = np.sqrt( pow(A_one_x, 2) + pow(A_two_x, 2));
  g_y = np.sqrt( pow(A_one_y, 2) + pow(A_two_y, 2));

  print "Found g"

  M_one_x = np.diag((1 + np.sign(diff_eigs_x) / huber_max_eigs)
                    + (1 - np.sign(diff_eigs_x) / huber_max_eigs))
  M_one_y = np.diag((1 + np.sign(diff_eigs_y) / huber_max_eigs)
                    + (1 - np.sign(diff_eigs_y) / huber_max_eigs))
  M_two_x = np.diag((1 + np.sign(diff_eigs_x) / huber_max_eigs)
                    - (1 - np.sign(diff_eigs_x) / huber_max_eigs))
  M_two_y = np.diag((1 + np.sign(diff_eigs_y) / huber_max_eigs)
                    - (1 - np.sign(diff_eigs_y) / huber_max_eigs))

  print "Found M1 and M2"

  Theta_x = np.diag( ((np.dot(np.sign(max_eigs_x),
                              (1 + np.sign(diff_eigs_x))) / g_x)
                      - (np.dot(np.sign(min_eigs_x),
                                (1 + np.sign(diff_eigs_x))) / g_x)))
  Theta_y = np.diag( ((np.dot(np.sign(max_eigs_y),
                              (1 + np.sign(diff_eigs_y))) / g_y)
                      - (np.dot(np.sign(min_eigs_y),
                                (1 + np.sign(diff_eigs_y))) / g_y)))

  print "Found Theta"

  # Find grad_J, used in the final optimization
  # gradJ = .5[T1' M1 T1 + T2' Theta T2 + T3' Theta T3] f + .5 [T1' M2 g]
  grad_J_x = .5 * np.dot((np.dot(np.dot(T_one_x.transpose(), M_one_x),
                                 T_one_x)
                          + np.dot(np.dot(T_two_x.transpose(), Theta_x),
                                   T_two_x)
                          + np.dot(np.dot(T_three_x.transpose(), Theta_x),
                                   T_three_x)),
                         f) + .5 * np.dot(np.dot(T_one_x.transpose(),
                                                   M_one_x), g_x);
  grad_J_y = .5 * np.dot((np.dot(np.dot(T_one_y.transpose(), M_one_y),
                                 T_one_y)
                          + np.dot(np.dot(T_two_y.transpose(), Theta_y),
                                   T_two_y)
                          + np.dot(np.dot(T_three_y.transpose(), Theta_y),
                                   T_three_y)),
                         f) + .5 * np.dot(np.dot(T_one_y.transpose(),
                                                   M_one_y), g_y);

  print "Found the gradient!"

  # Perform the optimization
  print grad_J_x.shape[0]
  print grad_J_y.shape[0]

  ###################
  # /Optimization iteration here
  ###################
