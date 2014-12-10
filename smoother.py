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


def costFunc(y, f, tau, JReg):
  imageCost = y - blurImage(f);
  # print .5 * np.sum(imageCost * imageCost) + tau * JReg;
  return .5 * np.square(np.linalg.norm(imageCost)) + tau * JReg;


# Apply some gaussian blur to this biznitch
def blurImage(src):
  ksize = (9, 9);
  sigma_x = 4;
  sigma_y = 4;
  blurimg = cv2.GaussianBlur(src, ksize, sigma_x,
                             sigma_y, cv2.BORDER_REPLICATE);
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
  # http://www.cs.uu.nl/docs/vakken/ibv/reader/chapter5.pdf
  partial_kernel = [[ 1, 0, -1],
                    [ 0, 0,  0],
                    [-1, 0,  1]];
  partial_kernel = np.asanyarray(partial_kernel);
  return cv2.filter2D(src, -1, partial_kernel);


# Do this element-wise through the full image
# http://bit.ly/163n9ci
def huberLoss(eigs, huberAlpha):
  # One may not be the best alpha term, but we'll start here.
  huberAlpha = 1;
  flag = eigs > huberAlpha;
  huberMask = np.greater(np.abs(eigs), huberAlpha);
  return (~huberMask * (0.5 * np.square(eigs))
          - huberMask * (huberAlpha * (0.5 * huberAlpha - abs(eigs))));


# http://bit.ly/1sewX7O
def calcGradJ(src, MOne, MTwo, Theta, g):
  superMOne = laplacian(laplacian(MOne));
  superThetaOne = negLaplacian(negLaplacian(Theta));
  superThetaTwo = partialDer(partialDer(Theta));
  superFKernel = np.dot(.5, (superMOne + superThetaOne + superThetaTwo) * src);
  superGKernel = np.dot(.5, laplacian(MTwo) * g);
  return (superFKernel + superGKernel);


##################################
# MAIN FUNCTION
##################################
if __name__ == '__main__':
  src = cv2.imread('./tree.jpg', 0)
  # Our y
  y = blurImage(src)
  # cv2.imshow("initial", y);
  # cv2.waitKey(0);
  # A good initial guess for our image
  f = y;
  maxiter = 100;
  huberAlpha = .5;
  backtrackAlpha = .02;
  backtrackBeta = .5;
  gradWeight = 5;  # Represents lambda in paper
  tau = 2; # No idea what this should be
  epsilon = 1e-5;

  for i in range(1, maxiter):
    print "Iteration: "
    print i
    #################
    # CALCULATE DERIVATIVES
    #################
    # AOne: The laplacian
    # http://bit.ly/12je4JY
    AOne = laplacian(f);
    # ATwo: The negative laplacian? Something like that.
    # It's the difference of two convolutions
    ATwo = negLaplacian(f);
    # AThree: Partial derivatives
    AThree = partialDer(f);

    #################
    # CALCULATE EIGENVALUES
    #################
    eigsMax = AOne + np.sqrt(np.square(ATwo) + np.square(AThree));
    eigsMin = AOne - np.sqrt(np.square(ATwo) + np.square(AThree));
    eigsMax[eigsMax == 0.0] = epsilon
    eigsMin[eigsMin == 0.0] = epsilon
    eigsDiff = np.abs(eigsMax) - np.abs(eigsMin);

    # Find JReg
    huberMax = huberLoss(eigsMax, huberAlpha);
    huberMin = huberLoss(eigsMin, huberAlpha);
    huberDiff = huberLoss(eigsDiff, huberAlpha);

    JReg = np.sum(.5 * (huberMax + huberMin + huberDiff));

    #################
    # CALCULATE VALUE
    #################

    val = costFunc(y, f, tau, JReg);
    print val;

    # # Find grad_JReg
    # All math here is element-wise
    g = np.sqrt(np.square(AOne) + np.square(ATwo));
    g[g == 0.0] = epsilon;

    MOne = (np.true_divide((1 + np.sign(eigsDiff)), huberMax)
            + np.true_divide((1 - np.sign(eigsDiff)), huberMin));
    MOne = np.nan_to_num(MOne);
    MTwo = (np.true_divide((1 + np.sign(eigsDiff)), huberMax)
            - np.true_divide((1 - np.sign(eigsDiff)), huberMin));
    MTwo = np.nan_to_num(MTwo);
    Theta = (np.true_divide(np.sign(eigsMax) * (1 + np.sign(eigsDiff)), g)
             - np.true_divide(np.sign(eigsMin) * (1 - np.sign(eigsDiff)), g));
    # Figure this is appropriate
    Theta = np.nan_to_num(Theta);
    # Find gradJReg, used in the final optimization. Just elementwise addition.
    gradJReg = calcGradJ(f, MOne, MTwo, Theta, g);
    grad = -blurImage(y - blurImage(f)) + np.dot((gradWeight * .5), gradJReg);

    delF = -grad;
    # Cutoff point, checked after step 1
    if np.linalg.norm(grad) <= epsilon:
      print "Iterations"
      print i;
      cv2.imwrite("./blurimg.jpg", f);

    # Else we keep going!
    # 2. Line Search
    t = 1

    # Search for optimal f value
    valStep = costFunc(y, f + np.dot(t, delF), tau, JReg);
    while np.any(valStep
                 > (val + backtrackAlpha * t
                    * np.dot(np.transpose(grad), delF))):
      t = t * backtrackBeta;
      valStep = costFunc(y, f + np.dot(t, delF), tau, JReg);

    # 3. Update to next step
    f = f + np.dot(t, delF);
    cv2.imwrite("./results/current iteration"+bin(i)+".jpg", f);
