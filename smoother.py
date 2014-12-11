# Purpose: to convolute an image and resmooth it out using
# a second-order smoothing term
# This paper was a great help:
# http://bit.ly/1uMjmsN
# OpenCV makes it easy to do these kinds of operations:
# http://bit.ly/1vZAjSU
# Handy write-up of rderivative-based image operations:
# http://bit.ly/1AhiZHc

# TODO: Might have to put in divisors for derivative kernels

import numpy as np
import scipy.linalg
import cv2
import time
import sys


def costFunc(y, f, tau, JReg):
  imageCost = y - blurImage(f);
  return .5 * np.square(np.linalg.norm(imageCost)) + tau * JReg;


def compare_img(orig_img, rect_img):
  img_diff = orig_img - rect_img
  img_norm = np.linalg.norm(img_diff, 2)
  return img_norm


def noisyImage(src):
  noise = np.zeros((src.shape[0], src.shape[1]));
  cv2.randn(noise, 0, 2);
  result = src + 10 * noise
  result[result<0] = 0;
  result[result>255] = 255;
  return result;

# Apply some gaussian blur to this biznitch
def blurImage(src):
  ksize = (5, 5);
  sigma_x = 2;
  sigma_y = 2;
  blurimg = cv2.GaussianBlur(src, ksize, sigma_x,
                             sigma_y, cv2.BORDER_REPLICATE);
  return blurimg;


def laplacian(src, boolTranspose):
  laplacian_kernel = [[0,  1,  0],
                      [1, -4,  1],
                      [0,  1,  0]];
  laplacian_kernel = np.asanyarray(laplacian_kernel);
  if boolTranspose:
    laplacian_kernel = laplacian_kernel.T;
  A_laplacian = cv2.filter2D(src, -1, laplacian_kernel);
  return A_laplacian;


def negLaplacian(src, boolTranspose):
  # Combine both kernels to get this
  double_kernel = [[0, -1,  0],
                   [1,  0,  1],
                   [0, -1,  0]];
  double_kernel = np.asanyarray(double_kernel);
  if boolTranspose:
    double_kernel = double_kernel.T;
  ANegLap = cv2.filter2D(src, -1, double_kernel);
  return ANegLap;


def partialDer(src, boolTranspose):
  # http://www.cs.uu.nl/docs/vakken/ibv/reader/chapter5.pdf
  partial_kernel = [[ 1, 0, -1],
                    [ 0, 0,  0],
                    [-1, 0,  1]];
  partial_kernel = np.asanyarray(partial_kernel);
  if boolTranspose:
    partial_kernel = partial_kernel.T;
  return cv2.filter2D(src, -1, partial_kernel);


# Do this element-wise through the full image
# http://bit.ly/163n9ci
def huberLoss(eigs, huberAlpha):
  # One may not be the best alpha term, but we'll start here.
  huberAlpha = .2;
  flag = eigs > huberAlpha;
  huberMask = np.greater(np.abs(eigs), huberAlpha);
  return (~huberMask * (0.5 * np.square(eigs))
          - huberMask * (huberAlpha * (0.5 * huberAlpha - abs(eigs))));


# http://bit.ly/1sewX7O
def calcGradJ(src, MOne, MTwo, Theta, g):
  superMOne = laplacian(laplacian(MOne, False), False);
  superThetaOne = negLaplacian(negLaplacian(Theta, False), False);
  superThetaTwo = partialDer(partialDer(Theta, False), False);
  superFKernel = .5 * (superMOne + superThetaOne + superThetaTwo) * src;
  superGKernel = .5 * laplacian(MTwo, False) * g;
  return (superFKernel + superGKernel);


##################################
# MAIN FUNCTION
##################################
if __name__ == '__main__':
  src = cv2.imread('./tree.jpg', 0);
  cv2.imwrite('./bwtree.jpg', src);
  # Our y
  y = noisyImage(src);
  cv2.imwrite('./noisyimg.jpg', y);
  # cv2.imshow("initial", y);
  # cv2.waitKey(0);
  # A good initial guess for our image
  f = np.zeros((y.shape[0], y.shape[1]));
  prev_f = y;
  maxiter = 1000;
  huberAlpha = .5;
  backtrackAlpha = .02;
  backtrackBeta = .5;
  gradWeight = .5;  # Represents lambda in paper
  norm_diff = 0;
  tau = 3; # No idea what this should be
  epsilon = .0001;

  for i in range(1, maxiter):
    # print "Iteration: "
    print i

    #################
    # CALCULATE DERIVATIVES
    #################
    # AOne: The laplacian
    # http://bit.ly/12je4JY
    AOne = laplacian(f, False);
    # ATwo: The negative laplacian? Something like that.
    # It's the difference of two convolutions
    ATwo = negLaplacian(f, False);
    # AThree: Partial derivatives
    AThree = partialDer(f, False);

    #################
    # CALCULATE EIGENVALUES
    #################
    eigsMax = AOne + np.sqrt(np.square(ATwo) + np.square(AThree));
    eigsMin = AOne - np.sqrt(np.square(ATwo) + np.square(AThree));
    eigsMax[eigsMax == 0.0] = epsilon;
    eigsMin[eigsMin == 0.0] = epsilon;
    eigsDiff = np.abs(eigsMax) - np.abs(eigsMin);

    # Find JReg
    huberMax = huberLoss(eigsMax, huberAlpha);
    huberMin = huberLoss(eigsMin, huberAlpha);
    huberDiff = huberLoss(eigsDiff, huberAlpha);
    JReg = np.sum(.5 * (huberMax + huberMin + huberDiff));
    # print JReg

    #################
    # CALCULATE VALUE
    #################

    current_cost = costFunc(y, f, tau, JReg);
    print "Current cost: ", current_cost;

    # Find grad_JReg
    # All math here is element-wise
    g = np.sqrt(np.square(AOne) + np.square(ATwo));
    g[g == 0.0] = epsilon;

    MOne = (np.true_divide((1 + np.sign(eigsDiff)), huberMax)
            + np.true_divide((1 - np.sign(eigsDiff)), huberMin));
    # MOne = np.nan_to_num(MOne);
    MTwo = (np.true_divide((1 + np.sign(eigsDiff)), huberMax)
            - np.true_divide((1 - np.sign(eigsDiff)), huberMin));
    # MTwo = np.nan_to_num(MTwo);
    Theta = (np.true_divide(np.sign(eigsMax) * (1 + np.sign(eigsDiff)), g)
             - np.true_divide(np.sign(eigsMin) * (1 - np.sign(eigsDiff)), g));
    # Figure this is appropriate
    # Theta = np.nan_to_num(Theta);
    # Find gradJReg, used in the final optimization. Just elementwise addition.
    gradJReg = calcGradJ(f, MOne, MTwo, Theta, g);
    # print "difference between src and y: ", compare_img(src, y);
    # print "difference between src and f: ", compare_img(src, f);
    grad = -blurImage(y - blurImage(f)) + (gradWeight * .5) * gradJReg;
    delF = -grad;
    # Cutoff point, checked after step 1
    # print "norm: ", np.linalg.norm(grad);
    # print "diff: ", np.linalg.norm(f - prev_f);
    if (np.linalg.norm(np.abs(grad)) <= epsilon
         or
        np.linalg.norm(f - prev_f) <= epsilon):
      print "Iterations"
      print i;
      cv2.imwrite("./rectimg.jpg", f);
      sys.exit();

    # Else we keep going!
    # 2. Line Search
    t = 1;

    while True:
      delta = np.dot(t, delF);
      # delta[np.abs(delta)<1e-5] = 0;
      # We're not in a pixel range
      if (np.all(f + delta <= 255) and
          np.all(f + delta >= 0)):
        break
      t = t * backtrackBeta;
      # print np.max((f + delta))
      # print np.min((f + delta))
      # print "T: ", t;

    # while True:
    #   delta = np.dot(t, delF);
    #   updated_cost = costFunc(y, f+delta, tau, JReg);
    #   cost_bound = (current_cost + backtrackAlpha * t
    #                 * np.dot(grad.T, delF));
    #   print "Cost diff: ", cost_bound - updated_cost;
    #   if updated_cost <= cost_bound:
    #     break
    #   t = t * backtrackBeta;
      # print t;


    # 3. Update to next step
    prev_f = f;
    f = f + np.dot(t, delF)
    f[f<0] = 0;
    f[f>255] = 255;

  # We didn't finish, but we ran out of iterations
  print "NO MORE iterations"
  print i;
  cv2.imwrite("./rectimg.jpg", f);
  # 164550856.545
