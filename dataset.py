import torch
import torch.nn
import os
from torch.utils.data import Dataset
from skimage import io
from skimage.transform import resize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SantaNoSanta(Dataset):
    def __init__(self, dog_root, cat_root, size, transform=None):
        self.transform = transform
        self.images_path, self.labels = self.label_builder(dog_root, cat_root)
        self.size = size

    @staticmethod
    def label_builder(dog, cat):
        dog_ = os.listdir(dog)
        dog_path = [dog + i for i in dog_]
        dog_label = [1] * len(dog_path)

        cat_ = os.listdir(cat)
        cat_path = [cat + i for i in cat_]
        cat_label = [0] * len(cat_path)

        return [*dog_path, *cat_path], [*dog_label, *cat_label]

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        image_path = self.images_path[item]
        image = io.imread(image_path)
        image = resize(image, (self.size, self.size))
        label = torch.tensor(int(self.labels[item]))
        if self.transform:
            image = self.transform(image)
        return (image, label)
