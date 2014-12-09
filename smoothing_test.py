# Test the goodness of the image transformation

import numpy as np
import cv2

def compare_img(orig_img, rect_img):
    img_diff = orig_img - rect_img
    print img_diff.shape
    img_norm = np.linalg.norm(img_diff, 2)
    return img_norm

def negLaplacianTest(src):
    # Showing that the kernels do the exact same thing. 
      # Horizontally
    horz_double_kernel = [[0,  0,  0],
                          [1, -2,  1],
                          [0,  0,  0]];
    horz_double_kernel = np.asanyarray(horz_double_kernel);
    A_horz = cv2.filter2D(src, -1, horz_double_kernel);
    # Vertically
    vert_double_kernel = [[0,  1,  0],
                          [0, -2,  0],
                          [0,  1,  0]];
    vert_double_kernel = np.asanyarray(vert_double_kernel);
    A_vert = cv2.filter2D(src, -1, vert_double_kernel);
    A_tot = A_horz - A_vert;
    # Combine both kernels to get this
    double_kernel = [[0, -1,  0],
                     [1,  0,  1],
                     [0, -1,  0]];
    double_kernel = np.asanyarray(double_kernel);
    ANegLap = cv2.filter2D(src, -1, double_kernel);
    return np.linalg.norm(ANegLap - A_tot);


def partialDerTest(src):
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
    A_final = cv2.filter2D(A_partial_horz, -1, vert_kernel);
    third_kernel = [[ 1, 0, -1],
                    [ 0, 0,  0],
                    [-1, 0, 1]];
    third_kernel = np.asanyarray(third_kernel);
    A_diff = cv2.filter2D(src, -1, third_kernel);
    return np.linalg.norm(A_diff - A_final);


if __name__ == '__main__':
    # Import as grayscale
    fn = './tree.jpg'
    blur_fn = './blurimg.jpg'
    rect_fn = './rectimg.jpg'
    img = cv2.imread(fn, 0)
    blur_img = cv2.imread(blur_fn, 0)
    rect_img = cv2.imread(rect_fn, 0)

    # Compare the data against ome another
    blur_diff = compare_img(img, blur_img)
    rect_diff = compare_img(img, rect_img)
    no_diff = compare_img(img, img)

    # Check results
    if no_diff != 0:
        print """ERROR: comparing the same images
        results in non-zero error."""
    if blur_diff > rect_diff:
        print """smoothing term is better than blurring term."""
        print "Smoothing: %f" % rect_diff
        print "Blurring: %f" % blur_diff
    if blur_diff <= rect_diff:
        print """ERROR: blurring term is better than smoothing term.
        How did you do that?"""
        print "Smoothing: %f" % rect_diff
        print "Blurring: %f" % blur_diff
