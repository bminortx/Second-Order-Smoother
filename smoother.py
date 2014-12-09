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
  return blurimg;


def laplacian(src):
  laplacian_kernel = [[0,  1,  0],
                      [1, -4,  1],
                      [0,  1,  0]];
  laplacian_kernel = np.asanyarray(laplacian_kernel);
  A_laplacian = cv2.filter2D(src, -1, laplacian_kernel);
  return A_laplacian;


def negLaplacian(src):
  # Combine both kernels to get this
  double_kernel = [[0, -1,  0],
                   [1,  0,  1],
                   [0, -1,  0]];
  double_kernel = np.asanyarray(double_kernel);
  ANegLap = cv2.filter2D(src, -1, double_kernel);
  return ANegLap;


def partialDer(src):
  # Convolute in horz, and use that result in convolution of vert
  horz_kernel = [[0,  0,  0],
                 [1,  0, -1],
                 [0,  0,  0]];
  horz_kernel = np.asanyarray(horz_kernel);
  A_partial_horz = cv2.filter2D(src, -1, horz_kernel);
  vert_kernel = [[0,  1,  0],
                 [0,  0,  0],
                 [0, -1,  0]];
  vert_kernel = np.asanyarray(vert_kernel);
  third_kernel = [[0,  1,  0],
                  [1,  0, -1],
                  [0, -1,  0]];
  return cv2.filter2D(A_partial_horz, -1, vert_kernel);


# Do this element-wise through the full image
# http://bit.ly/163n9ci
def huberLoss(eigs):
  # One may not be the best alpha term, but we'll start here.
  alpha = 1;
  flag = eigs > alpha;
  huberMask = np.greater(np.abs(eigs), alpha);
  return (~huberMask * (0.5 * np.square(eigs))
          - huberMask * (alpha * (0.5 * alpha - abs(eigs))));


# http://bit.ly/1sewX7O
def calcGradJ(src, rows, cols, MOne, MTwo, Theta, g):
  ##############
  # 1. Compute kernel on f
  for j in range(1, rows - 1):
    # Access the following rows
    previous = src[j - 1, :];
    current = src[j, :];
    next = src[j + 1, :];
    fOutput = np.zeros((rows, cols));
    laplacian_kernel = np.array([[0,  1,  0],
                                 [1, -4,  1],
                                 [0,  1,  0]]);
    double_kernel = [[0, -1,  0],
                     [1,  0,  1],
                     [0, -1,  0]];
    for i in range(1, cols - 1):
      # TODO: Place custom kernel here
      kernelTOne = np.dot(laplacian_kernel.T,
                          np.dot(MOne[j, i], laplacian_kernel));
      kernelTOne = np.dot(laplacian_kernel.T,
                          np.dot(MOne[j, i], laplacian_kernel));
      # Endpoint is weird for numpy
      superpixel = src[j - 1 : j + 2, i - 1 : i + 2];
      fOutput[j, i] = np.sum(np.dot(kernelTOne, superpixel));

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
  # AOne: The laplacian
  # http://bit.ly/12je4JY
  AOne = laplacian(blurimg);

  # ATwo: The negative laplacian? Something like that.
  # It's the difference of two convolutions
  ATwo = negLaplacian(blurimg);
  cv2.imwrite("ATwo.png", ATwo)

  # AThree: Partial derivatives
  AThree = partialDer(blurimg);
  cv2.imwrite("AThree.png", AThree)

  # Well that was stupid easy.

  #################
  # CALCULATE EIGENVALUES
  #################

  eigsMax = AOne + np.sqrt(np.square(ATwo) + np.square(AThree));
  eigsMin = AOne - np.sqrt(np.square(ATwo) + np.square(AThree));
  eigsDiff = eigsMax - eigsMin;
  print "Found max and min eigenvalues"

  # Find JReg
  huberMax = huberLoss(eigsMax);
  huberMin = huberLoss(eigsMin);
  huberDiff = huberLoss(eigsDiff);
  print "Found Huber norm of eigen pairs"

  JReg = np.sum(.5 *
                (huberMax + huberMin + huberDiff));
  print "Found JReg, the regularization term. It's value is"
  print JReg

  # # Find grad_JReg
  # All math here is element-wise
  g = np.sqrt(np.square(AOne) + np.square(ATwo));
  print "Shape of g: "
  print g.shape

  MOne = (np.true_divide((1 + np.sign(eigsDiff)), huberMax)
           + np.true_divide((1 - np.sign(eigsDiff)), huberMax));
  MOne = np.nan_to_num(MOne);
  MTwo = (np.true_divide((1 + np.sign(eigsDiff)), huberMax)
           - np.true_divide((1 - np.sign(eigsDiff)), huberMax));
  MTwo = np.nan_to_num(MTwo);
  print "Found M1 and M2"

  Theta = (np.true_divide(np.sign(eigsMax) * (1 + np.sign(eigsDiff)), g)
           - np.true_divide(np.sign(eigsMin) * (1 + np.sign(eigsDiff)), g));
  # Figure this is appropriate
  Theta = np.nan_to_num(Theta);
  print "Found Theta. Shape:"
  print Theta.shape

  # # Find gradJReg, used in the final optimization. Just elementwise addition.
  (fOutput, gOutput) = calcGradJ(blurimg, rows, cols, MOne, MTwo, Theta, g);
  gradJReg = fOutput + gOutput;
  print "Found the gradient for the regularizer!"

  ###################
  # /Optimization iteration here
  ###################
