from constants import COLOR_CODES_BY_CLASS as COLOR_CODES_BY_CLASS
from skimage import io
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

num_classes = 1

class ImageLoader:
    @staticmethod
    def load_image(path:str):
        img = cv2.imread(path)
        
        return np.transpose(img, (2, 0, 1))
    
    @staticmethod
    def load_image_without_transpose(path:str):
        img = cv2.imread(path)
        
        return img

    @staticmethod
    def load_images(path, image_names:list):
        return np.array([ImageLoader.load_image(path + "/" + image_name) for image_name in image_names])

    @staticmethod
    def load_image_names(directory:str, extension:str):
        return np.array([file_name for file_name in os.listdir(directory) if file_name.endswith(extension)])


def _normalize_image(img):
    return img / 255.0

class ImageDataset(Dataset):    
    def __init__(self, root_directory_input, root_directory_output, image_names_input, image_names_output):
        self.root_directory_input = root_directory_input
        self.root_directory_output = root_directory_output
        self.image_names_input = image_names_input
        self.image_names_output = image_names_output

    def __getitem__(self, idx):
        img_name_input = os.path.join(self.root_directory_input, self.image_names_input[idx])
        img_name_output = os.path.join(self.root_directory_output, self.image_names_output[idx])

        image_input = io.imread(img_name_input)
        image_input = _normalize_image(image_input)
        image_input = np.transpose(image_input, (2, 0, 1))

        image_output = io.imread(img_name_output)

        label = np.expand_dims(image_output.astype('float32'), axis=0)

        return img_name_input.split("\\")[-1][:-4], image_input, label

    def __len__(self):
        return len(self.image_names_input)