import cv2
import matplotlib.pylab as plt
import numpy as np
from itertools import combinations, product, permutations
import seaborn as sns
sns.set()
import os


class ImgMorph:
    def __init__(self, img, name=None, gray_scaled=False):
        self.img = img
        self.name = name
        self.gray_scaled = gray_scaled
        self.histogram = None

    def shape(self):
        return self.img.shape

    def change_color_space(self, tcs):
        return ImgMorph(cv2.cvtColor(self.img, tcs), gray_scaled=self.gray_scaled)

    def gray(self):
        return ImgMorph(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), gray_scaled=True)

    def normalize(self, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX):
        return ImgMorph(cv2.normalize(self.img, None, alpha=0, beta=255, 
                        norm_type=norm_type), gray_scaled=self.gray_scaled)

    def crop(self, x, y, dx, dy):
        return ImgMorph(self.img[y - dy:y + dy, x - dx:x + dx], gray_scaled=self.gray_scaled)

    def filter2D(self, kernel):
        return ImgMorph(cv2.filter2D(self.img, -1, kernel))

    def equalize(self):
        if self.gray_scaled:
            return ImgMorph(cv2.equalizeHist(self.img), gray_scaled=self.gray_scaled)
        else:
            channels = cv2.split(self.img)
            channels_ = []
            for chn in channels:
                channels_.append(cv2.equalizeHist(chn))
            out = cv2.merge(channels_)
            return ImgMorph(out, gray_scaled=self.gray_scaled)
            
    def threshold(self, t=0, upper=255, kind="0"):
        if kind == "0":
            c = 0
        elif kind.upper() == "THRESH_BINARY_INV":
            c = cv2.THRESH_BINARY_INV
        return ImgMorph(cv2.threshold(self.img, t, upper, c)[1], gray_scaled=self.gray_scaled)

    def open(self, kernel, iterations: int = 1):
        return ImgMorph(cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel, iterations=iterations), gray_scaled=self.gray_scaled)

    def close(self, kernel, iterations: int = 1):
        return ImgMorph(cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel, iterations=iterations), gray_scaled=self.gray_scaled)

    def dilate(self, kernel):
        return ImgMorph(cv2.dilate(self.img, kernel, 0), gray_scaled=self.gray_scaled)

    def find_contours(self):
        return cv2.findContours(self.img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def __add__(self, other):
        return ImgMorph((np.logical_or(self.img, other.img) * 255).astype(np.uint8))

    def calculate_histogram(self, normalize=True, num_bins=256):
        color = ("k",) if self.gray_scaled else ('b','g','r')
        histogram = []
        for i, col in enumerate(color):
            histr = cv2.calcHist([self.img], [i], None, [num_bins], [0, 256], )
            if normalize:
               histr = cv2.normalize(histr, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            histogram.append((histr, col))

        self.histogram = histogram

    def compare_histogram(self, other, distance, channel=0):
        """
        distance:
            cv2.HISTCMP_CORREL
            cv2.HISTCMP_CHISQR
            cv2.HISTCMP_INTERSECT
            cv2.HISTCMP_BHATTACHARYYA
            cv2.HISTCMP_CHISQR_ALT
            cv2.HISTCMP_KL_DIV
        """
        if self.histogram is None:
            self.calculate_histogram()
        if other.histogram is None:
            other.calculate_histogram()

        return cv2.compareHist(self.histogram[channel][0], other.histogram[channel][0], distance)

    @property
    def picture(self):
        self.show_picture(3)

    def show_picture(self, size):
        plt.figure(figsize=(size, size))
        ax = plt.axes([0, 0, 1, 1], frameon=False)
        ax.set_axis_off()
        if self.gray_scaled:
            plt.imshow(self.img, cmap="gray")
        else:
            plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))

    def show_histogram(self, size=5, normalize=True):
        fig, ax = plt.subplots(1, 2, figsize=(size * 2, size))

        ax[0].set_axis_off()
        ax[0].imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))

        if self.histogram is None:
            self.calculate_histogram(normalize=normalize)

        for histr, color in self.histogram:
            ax[1].plot(histr, color=color)

        ax[1].set_xlim([0, 256])
        plt.show()