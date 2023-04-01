import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.misc
import os,sys
import cv2
import mkl_fft
#import pymp

file  = sys.argv[1]
img = cv2.imread(file)


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()
    
def deblur_channel(img,cor_v=0.055):
    channel = []
    final = []
    gaussian_3 = cv2.GaussianBlur(img, (9,9), 10.0)
    img = cv2.addWeighted(img, 1, gaussian_3, -0.01, 2)
    (b,g,r) =  cv2.split(img)
    channel.append(r)
    channel.append(g)
    channel.append(b)

    for index in range(0, 3):
        f = np.fft.fft2(channel[index])
        #f = mkl_fft.fft2(channel[index], shape=None, axes=(-2,-1), overwrite_x=False)
        fshift = np.fft.fftshift(f)
        rows, cols = channel[index].shape
        crow,ccol = int(rows/2) , int(cols/2)
        # remove the low frequencies by masking with a rectangular window of size 60x60
        # High Pass Filter (HPF)
        new = fshift[crow-int(crow*(1-cor_v)):crow+int(crow*(1-cor_v)), ccol-int(ccol*(1-cor_v)):ccol+int(ccol*(1-cor_v))]
        #fshift[crow-1:crow+1, ccol-1:ccol+1] = 0
        # shift back (we shifted the center before)
        f_ishift = np.fft.ifftshift(new)
        # inverse fft to get the image back
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        final.append(img_back)
    bgr = cv2.merge([final[2],final[1],final[0]])	
    out = cv2.convertScaleAbs(bgr)
    return out

cache = deblur_channel(img)
cv2.imwrite(file, cache)
