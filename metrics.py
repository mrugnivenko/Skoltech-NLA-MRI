import numpy
import numpy as np
import math
from scipy.ndimage import convolve
# TODO 4 more metrics


def psnr(img1, img2):
    """
    Measure of Peak Signal-to-Noise Ratio (PSNR) [22] measures the ratio between the maximum possible value of a signal
    and the power of distorting noise that affects the quality of its representation. In Fig. 5-a), while the value of (μ)
    is 0.65, PSNR value of proposed method is the highest as compared to the other methods and it is up to 12 dB.
    This involves better quality of the image as well as best noise reduction.
    """

    img1 = (img1 - img1.min())/(img1.max() - img1.min())
    img2 = (img2 - img2.min())/(img2.max() - img2.min())
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100000
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))



def qrcm(img1, img2):
    """
    Measure of Quality-aware Relative Contrast Measure (QRCM) [23] gives idea about contrast measure and image quality
    together. QRCM penalizes the contrast changes when there is a significant difference between the gradients of
    original and enhanced images. This is happened generally when there are visual distortions on the processed image.
    Thus, QRCM does not only measure the relative change of contrast but also takes the distortion introduced on the
    enhanced image relative to the considered original image.
    Negative QRCM values indicate that considered contrast enhancement algorithm distorted enhanced image
     as comparing to the original one.
    """
    eps = 1e-8
    img1 = (img1 - img1.min())/(img1.max() - img1.min()) * 255
    img2 = (img2 - img2.min())/(img2.max() - img2.min()) * 255
    
    kernel_average = 1 / 9 * np.ones((3, 3))
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype='float32')
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype='float32')
    
    img1 = convolve(img1, kernel_average)
    grad_x = convolve(img1, kernel_x) 
    grad_y = convolve(img1, kernel_y) 
    G1 = np.hypot(grad_x, grad_y)
    
    img2 = convolve(img2, kernel_average)    
    grad_x = convolve(img2, kernel_x)
    grad_y = convolve(img2, kernel_y) 
    G2 = np.hypot(grad_x, grad_y)
    
    G12 = (G2 - G1) / (G2 + G1 + eps)
    w1 = G1 / G1.sum()
    RCM = (G12 * w1).sum()
    T = 255 / np.sqrt(2)
    GMS = (2 * G1 * G2 + T) / (G1**2 + G2**2 + T)
    mu = GMS.mean()
    w2 = 1 / (1 + G1)
    Q = 1 - 1 / img1.size * (np.abs(GMS - mu) * w2).sum()
    if RCM >= 0:
        QRCM = RCM * Q
    else:
        QRCM = (1 + RCM) * Q - 1
    return QRCM
