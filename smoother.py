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
import sys


def costFunc(y, f, tau, JReg):
  imageCost = y - convolve(f);
  cost = .5 * np.square(np.linalg.norm(imageCost)) + tau * JReg;
  return cost


# Calculates the ISNR
def compare_img(orig_img, bad_img, rect_img):
  e1 = orig_img - bad_img;
  e2 = orig_img - rect_img;
  E1 = np.mean(np.square(e1));
  E2 = np.mean(np.square(e2));
  result = 10 * np.log(E1 / E2) / np.log(10)
  return result


def convolve(src):
  # Start with identity
  # convolve_kernel = [[0, 0, 0],
  #                    [0, 1, 0],
  #                    [0, 0, 0]];
  # convolve_kernel = np.asanyarray(convolve_kernel);
  # A_convolve = cv2.filter2D(src, -1, convolve_kernel);
  A_convolve = noisyImage(src);
  return A_convolve;

def noisyImage(src):
  noise = np.zeros((src.shape[0], src.shape[1]));
  cv2.randn(noise, 0, 10);
  result = src + noise
  result[result<0] = 0;
  result[result>255] = 255;
  return result;

# Apply some gaussian blur to this biznitch
def blurImage(src):
  ksize = (3, 3);
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
  flag = eigs > huberAlpha;
  huberMask = np.greater(np.abs(eigs), huberAlpha);
  return (~huberMask * (0.5 * np.square(eigs))
          - huberMask * (huberAlpha * (0.5 * huberAlpha - abs(eigs))));


# http://bit.ly/1sewX7O
def calcGradJ(src, MOne, MTwo, Theta, g):
  superMOne = laplacian(laplacian(MOne));
  superThetaOne = negLaplacian(negLaplacian(Theta));
  superThetaTwo = partialDer(partialDer(Theta));
  superFKernel = .5 * (superMOne + superThetaOne + superThetaTwo) * src;
  superGKernel = .5 * laplacian(MTwo) * g;
  return (superFKernel + superGKernel);


##################################
# MAIN FUNCTION
##################################
if __name__ == '__main__':
  src = cv2.imread('./tree.jpg', 0);
  cv2.imwrite('./bwtree.jpg', src);
  # Our y
  y = blurImage(noisyImage(src));
  cv2.imwrite('./noisyimg.jpg', y);
  # cv2.imshow("initial", y);
  # cv2.waitKey(0);
  # A good initial guess for our image
  f = np.zeros((y.shape[0], y.shape[1]));
  prev_f = y;
  prev_cost = 1e50;
  maxiter = 100;
  huberAlpha = 1;
  backtrackAlpha = .02;
  backtrackBeta = .5;
  backtrackGamma = .5;
  # gradWeight = .5;  # Represents lambda in paper
  norm_diff = 0;
  epsilon = .0000001;
  ISNR = compare_img(src, y, f);

  for i in range(1, maxiter):
    gradWeight = 1;
    tau = 5; # No idea what this should be
    # print "Iteration: "
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
    eigsMax = (AOne + np.sqrt(np.square(ATwo) + np.square(2 * AThree)));
    eigsMin = (AOne - np.sqrt(np.square(ATwo) + np.square(2 * AThree)));
    eigsMax[eigsMax <= 0.0] = epsilon;
    eigsMin[eigsMin <= 0.0] = epsilon;
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
    MOne = (((1 + np.sign(eigsDiff)) / huberMax)
            + ((1 - np.sign(eigsDiff)) / huberMin));
    MTwo = (((1 + np.sign(eigsDiff)) / huberMax)
            - ((1 - np.sign(eigsDiff)) / huberMin));
    Theta = (((np.sign(eigsMax) * (1 + np.sign(eigsDiff))) / g)
             - ((np.sign(eigsMin) * (1 - np.sign(eigsDiff))) / g));
    # Find gradJReg, used in the final optimization.
    gradJReg = calcGradJ(f, MOne, MTwo, Theta, g);
    # print "Grad J Reg: ", gradJReg;
    ISNR = compare_img(src, y, f);
    print "ISNR: ", ISNR
    # print "difference between src and f: ", compare_img(src, f);
    img_term = -convolve(y - convolve(f));
    reg_term = (tau * .5) * gradJReg;
    grad = img_term + reg_term;
    delF = -grad;

    # Cutoff point, checked after step 1
    if (np.linalg.norm(grad) <= epsilon  or
      np.square(np.linalg.norm(f - prev_f)) <= epsilon):
      print "Iterations"
      print i;
      ISNR = compare_img(src, y, f);
      print "New ISNR: ", ISNR
      cv2.imwrite("./rectimg.jpg", f);
      sys.exit();

    # Else we keep going!
    # 2. Line Search
    # t = .02;
    ii = 0;
    while True:
      img_term = -convolve(y - convolve(f + delF));
      reg_term = (gradWeight * .5) * gradJReg;
      grad = img_term + reg_term;
      delF = -grad;
      newCost = costFunc(y, f + delF, tau, JReg);
      if (newCost < current_cost or ii > 100):
        break
      gradWeight = gradWeight * backtrackBeta;
      # print "weight: ", gradWeight
      ii = ii + 1

    t = 1;
    if ii > 100:
      while True:
        delta = delF * t
        newCost = costFunc(y, f + delta, tau, JReg);
        if (newCost < current_cost):
          break
        t = t * backtrackGamma;
        # print "t: ", t


    # 3. Update to next step
    prev_f = f;
    f = f + delF * t;
    f[f<0] = 0;
    f[f>255] = 255;

  # We didn't finish, but we ran out of iterations
  print "NO MORE iterations"
  print i;
  cv2.imwrite("./rectimg.jpg", f);
  # 164550856.545
