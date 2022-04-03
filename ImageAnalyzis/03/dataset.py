import os
from imgmorph import ImgMorph
import cv2


class MyDataset:
    def __init__(self, path_to_dataset, **kwargs):
        self.pic_names = os.listdir(path_to_dataset)
        self.path_base = path_to_dataset
    
    def __getitem__(self, index):
        name = self.pic_names[index]
        path = self.path_base + "/" + name
        return ImgMorph(cv2.imread(path), name=name)
    
    def __len__(self):
        return len(self.pic_names)
