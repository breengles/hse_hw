import numpy as np


class LshCode():
    def __init__(self, centroids, descriptors, length):
        self.centroids = centroids
        self.length = length
        points = (descriptors - centroids).copy()
        line_length = points.shape[1]
        self.norm_array = np.random.uniform(-100, 100, (self.length, line_length))
        self.norm_array /= np.linalg.norm(self.norm_array, axis=1).reshape(-1,1)
        self.d_array = np.random.normal(0, np.median(np.std(points, 0)), self.length)
        self.lsh_codes = np.zeros((self.length, len(points)))
        for i in range(self.length):
            self.lsh_codes[i] = points.dot(self.norm_array[i]) + self.d_array[i] > 0
        self.lsh_codes = self.lsh_codes.T
        
    def get_norm(self):
        return self.norm_array
    
    def get_d(self):
        return self.d_array
    
    def get_lsh_codes(self):
        return self.lsh_codes
    
    def create_lsh_code(self, point):
        p = (point -  self.centroids).copy()
        lsh_code = np.zeros(self.length)
        for i in range(self.length):
            lsh_code[i] = p.dot(self.norm_array[i]) + self.d_array[i] > 0
        return lsh_code
    