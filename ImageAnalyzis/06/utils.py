import matplotlib.pyplot as plt
import numpy as np
import cv2


def grid(array, ncols=3):  
    # https://stackoverflow.com/questions/42040747/more-idiomatic-way-to-display-images-in-a-grid-with-numpy
    nindex, height, width, intensity = array.shape
    nrows = nindex // ncols
    return (array.reshape(nrows, ncols, height, width, intensity).swapaxes(1,2).reshape(height*nrows, width*ncols, intensity))

def show(img, gray=False, size=3):
    plt.figure(figsize=(size, size))
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    if gray:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
        
def gabour_bank(size, phi_bins, scale_bins, max_scale, min_scale, sigma_scale=0.5, psi=0):
    """
    Parameters:
    size (tuple|int): Size or radius of filters.
    phi_bins (int): Number of angle beens of bank
    scale_bins (int): Number of scales betwwen max_scale and min_scale
    max_scale (float): Max frequency covered by bank
    min_scale (float): Min frequency covered by bank
    Returns:
    list: a list of filters
    """

    # if isinstance(size, int):
    #     size_ = (size, size)
    # else:
    #     size_ = size

    lambdas = np.linspace(min_scale, max_scale, scale_bins)
    thetas = np.linspace(0, np.pi, phi_bins, endpoint=False)
    # filters = np.zeros((scale_bins, phi_bins, *size_))
    # filters = np.zeros((phi_bins, scale_bins, *size_))
    filters = []
    for j, theta in enumerate(thetas):
        for i, lam in enumerate(lambdas):
            size_ = int(1.5 * lam)
            sigma = sigma_scale * lam
            kernel = cv2.getGaborKernel((size_, size_), sigma, theta, lam, gamma=1, psi=psi)
            kernel[kernel > 0] /= np.sum(kernel[kernel > 0])
            kernel[kernel < 0] /= np.abs(np.sum(kernel[kernel < 0]))
            filters.append(np.copy(kernel))
            # filters[i, j] = np.copy(kernel)
    return filters