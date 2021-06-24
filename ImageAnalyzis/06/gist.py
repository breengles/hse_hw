import numpy as np
import cv2


class Gist():
    def __init__(self, filters, phi_bins=6, scale_bins=5, window_size = 4):
        self.win = window_size
        self.w = phi_bins
        self.h = scale_bins
        self.filters = filters

    # def get_gist_descriptor(self, image) -> np.ndarray:
    #     h_spl = np.array_split(image, self.win, axis=0)
    #     w_spl = [np.array_split(it, self.win, axis=1) for it in h_spl]
    #     descriptor = np.array([[it.mean() for it in row] for row in w_spl]).flatten()
    #     return descriptor


    def get_gist_descriptor(self, image):
        h, w, c = image.shape
        
        res = np.zeros((self.h * h, self.w * w, c), dtype = 'float')
        res1 = np.zeros((self.h * self.win, self.w * self.win, c), dtype = 'float')

        descriptor = []

        for i in range(self.h):
            for j in range(self.w):
                temp = cv2.filter2D(image, -1, self.filters[self.w * i + j])
                res[i * h:(i + 1) * h, j * w:(j + 1) * w] = \
                    cv2.normalize(temp, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                for y in range(self.win):
                    for x in range(self.win):
                        res1[i * self.win + y, j * self.win + x] = \
                            np.mean(res[i * h + int(y * h / self.win):i * h + int((y + 1) * h / self.win), 
                                        j * w + int(x * w / self.win):j * w + int((x + 1) * w / self.win)], 
                                    axis = (0, 1))
                descriptor.extend(res1[i * self.win:(i + 1) * self.win, 
                                       j * self.win:(j + 1) * self.win].flatten())
        return np.array(descriptor)
