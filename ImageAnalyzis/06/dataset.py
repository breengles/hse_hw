import cv2


class MyDataset:
    def __init__(self, imgs_name):
        self.photo_names = imgs_name
    
    def __getitem__(self, index):
        return cv2.imread(self.photo_names[index])
    
    def __len__(self):
        return len(self.photo_names)
    