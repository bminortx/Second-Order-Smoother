# Purpose: to convolute an image and resmooth it out using
# a second-order smoothing term
# This paper was a great help:
# http://bit.ly/1uMjmsN
# OpenCV makes it easy to do these kinds of operations:
# http://bit.ly/1vZAjSU
# Handy write-up of rderivative-based image operations:
# http://bit.ly/1AhiZHc

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


def laplacian(src):
  laplacian_kernel = [[0,  1,  0],
                      [1, -4,  1],
                      [0,  1,  0]];
  laplacian_kernel = np.asanyarray(laplacian_kernel);
  A_laplacian = cv2.filter2D(blurimg, -1, laplacian_kernel);
  return A_laplacian;


def negLaplacian(src):
  # Horizontally
  horz_double_kernel = [[0,  0,  0],
                        [1, -2,  1],
                        [0,  0,  0]];
  horz_double_kernel = np.asanyarray(horz_double_kernel);
  A_horz = cv2.filter2D(blurimg, -1, horz_double_kernel);
  # Vertically
  vert_double_kernel = [[0,  1,  0],
                        [0, -2,  0],
                        [0,  1,  0]];
  vert_double_kernel = np.asanyarray(vert_double_kernel);
  A_vert = cv2.filter2D(blurimg, -1, vert_double_kernel);
  return A_horz - A_vert;


def partialDer(src):
  # Convolute in horz, and use that result in convolution of vert
  horz_kernel = [[0,  0,  0],
                 [1,  0, -1],
                 [0,  0,  0]];
  horz_kernel = np.asanyarray(horz_kernel);
  A_partial_horz = cv2.filter2D(blurimg, -1, horz_kernel);
  vert_kernel = [[0,  1,  0],
                 [0,  0,  0],
                 [0, -1,  0]];
  vert_kernel = np.asanyarray(vert_kernel);
  return cv2.filter2D(A_partial_horz, -1, vert_kernel);

# Elegantly programmed norm from http://bit.ly/1xuPoqn
# Do this element-wise through the full image
# http://bit.ly/163n9ci
def huberLoss(eigs):
  # One may not be the best alpha term, but we'll start here.
  alpha = 1;
  flag = eigs > alpha;
  huberMask = np.greater(np.abs(eigs), alpha);
  return (~huberMask * (0.5 * eigs ** 2)
          - huberMask * (alpha * (0.5 * alpha - abs(eigs))));


# http://bit.ly/1sewX7O
def calcGradJ(src, rows, cols, M1, M2, Theta, g):
  ##############
  # 1. Compute kernel on f
  for j in range(1, rows - 1):
    # Access the following rows
    previous = src[j - 1, :];
    current = src[j, :];
    next = src[j + 1, :];
    fOutput = np.zeros((rows, cols));
    for i in range(1, cols - 1):


      
      # TODO: Place custom kernel here
      fOutput[j, i] = (5 * current[i] - current[i - 1]
                      - current[i + 1] - previous[i] - next[i]);


      

  # Zero out the other rows
  fOutput[0, :] = np.zeros((1, cols));
  fOutput[rows - 1, :] = np.zeros((1, cols));
  fOutput[:, 0] = np.zeros((rows));
  fOutput[:, cols - 1] = np.zeros((rows));

  ##############
  # 2. Compute kernel on g
  for j in range(1, rows - 1):
    # Access the following rows
    previous = src[j - 1, :];
    current = src[j, :];
    next = src[j + 1, :];
    gOutput = np.zeros((rows, cols));
    for i in range(1, cols - 1):


      
    # TODO: Place custom kernel here
      gOutput[j, i] = (5 * current[i] - current[i - 1]
                      - current[i + 1] - previous[i] - next[i]);


      

  # Zero out the other rows
  gOutput[0, :] = np.zeros((1, cols));
  gOutput[rows - 1, :] = np.zeros((1, cols));
  gOutput[:, 0] = np.zeros((rows));
  gOutput[:, cols - 1] = np.zeros((rows));

  
  return (fOutput, gOutput);



##################################
# MAIN FUNCTION
##################################
if __name__ == '__main__':
  fn = cv2.imread('./tree.jpg', 0)
  blurimg = blurImage(fn)
  rows, cols = blurimg.shape

  #################
  # CALCULATE DERIVATIVES
  #################
  # A_one: The laplacian
  # http://bit.ly/12je4JY
  A_one = laplacian(blurimg);

  # A_two: The negative laplacian? Something like that.
  # It's the difference of two convolutions
  A_two = negLaplacian(blurimg);
  cv2.imwrite("A_two.png", A_two)

  # A_three: Partial derivatives
  A_three = partialDer(blurimg);
  cv2.imwrite("A_three.png", A_three)

  # Well that was stupid easy.

  #################
  # CALCULATE EIGENVALUES
  #################

  max_eigs = A_one + np.sqrt(np.square(A_two) + np.square(A_three));
  min_eigs = A_one - np.sqrt(np.square(A_two) + np.square(A_three));
  diff_eigs = max_eigs - min_eigs;

  print "Found max and min eigenvalues"

  # Find J_new
  huber_max_eigs = huberLoss(max_eigs)
  huber_min_eigs = huberLoss(min_eigs)
  huber_diff_eigs = huberLoss(diff_eigs)
  print huber_max_eigs.shape
  print huber_min_eigs.shape
  print huber_diff_eigs.shape

  print "Found Huber norm of eigen pairs"

  J_new = np.sum(.5 * (huber_max_eigs + huber_min_eigs + huber_diff_eigs));

  print "Found J_new, the regularization term. It's value is"
  print J_new

  # # Find grad_J_new
  # All math here is element-wise
  g = np.sqrt(np.square(A_one) + np.square(A_two))
  print "Shape of g: "
  print g.shape

  M_one = np.diagflat(
    np.diagonal(np.divide((1 + np.sign(diff_eigs)), huber_max_eigs)
                + np.divide((1 - np.sign(diff_eigs)), huber_max_eigs)))
  M_two = np.diagflat(
    np.diagonal(np.divide((1 + np.sign(diff_eigs)), huber_max_eigs)
                - np.divide((1 - np.sign(diff_eigs)), huber_max_eigs)))

  print "Found M1 and M2"

  Theta = np.diagflat(
    np.diagonal(np.divide(np.sign(max_eigs) * (1 + np.sign(diff_eigs)), g)
                - np.divide(np.sign(min_eigs) * (1 + np.sign(diff_eigs)), g)))
  # Figure this is appropriate
  Theta = np.nan_to_num(Theta)

  print "Shape of Theta: "
  print Theta.shape

  # # Find grad_J, used in the final optimization

  (fOutput, gOutput) = calcGradJ(blurimg, rows, cols, M1, M2, Theta, g);


  # print "Found the gradient!"

  # # Perform the optimization
  # print grad_J_x.shape[0]
  # print grad_J_y.shape[0]

  ###################
  # /Optimization iteration here
  ###################
