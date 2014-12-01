# Test the goodness of the image transformation

import numpy as np
import cv2

def compare_img(orig_img, rect_img):
    img_diff = orig_img - rect_img
    print img_diff.shape
    img_norm = np.linalg.norm(img_diff, 2)
    return img_norm

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
