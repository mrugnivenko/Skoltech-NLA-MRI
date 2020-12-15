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
    kernel_x = 1 / 3 * np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype='float32')
    kernel_y = 1 / 3 * np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype='float32')
    
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

def ssim(img1_orig, img2_orig):
    """
    Structure similarity index measurement (SSIM) [24] is a metric based on 
    measuring the similarity between two images. The SSIM can take values in  
    range. Higher value suggests the better preservation of image structures. 
    """
    img1_orig = (img1_orig - img1_orig.min())/(img1_orig.max() - img1_orig.min()) * 255
    img2_orig = (img2_orig - img2_orig.min())/(img2_orig.max() - img2_orig.min()) * 255
    res = np.zeros(((img1_orig.shape[0]//8), (img1_orig.shape[1]//8)))
    for i in range(img1_orig.shape[0]//8):
        for j in range(img1_orig.shape[1]//8):
            img1 = img1_orig[8*i:8*(i+1), 8*j:8*(j+1)]
            img2 = img2_orig[8*i:8*(i+1), 8*j:8*(j+1)]            
            mu_x = img1.mean()
            mu_y = img2.mean() 
            sigma_x = img1.std()
            sigma_y = img2.std()
            sigma_xy = 1/img1.size*(((img1-img1.mean())*(img2-img2.mean())).sum())
            L = 255
            c1 = (0.01*L)**2
            c2 = (0.03*L)**2
            res[i, j] = ((2*mu_x*mu_y + c1)*(2*sigma_xy + c2))/((mu_x**2 + mu_y**2 + c1)*(sigma_x**2 + sigma_y**2 + c2))
    return res.mean()

def EME(image, k1 = 4, k2 = 4): 
    image = image.astype('float64')
    if image.min() == 0:
        image += 0.01
        image = image/image.max()
    result = 0
    block_i_size = image.shape[0]//k1
    block_j_size = image.shape[1]//k2
    
    for i in range(k1):
        for j in range(k2):
            block = image[i*block_i_size:(i+1)*block_i_size, j*block_j_size:(j+1)*block_j_size]
            result += 20*np.log(block.max()/block.min())
    return result/(k1*k2)

def AMBE(img1,img2):
    return abs(np.mean(img1)-np.mean(img2))