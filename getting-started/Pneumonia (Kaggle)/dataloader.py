import os
import random
from matplotlib.pyplot import imread
import PIL as pil
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

def absoluteFilePaths(directory):
    paths = []
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            paths.append(os.path.abspath(os.path.join(dirpath, f)))
    return paths


class ChestXRayImageData(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor()
        ])
        self.imList = absoluteFilePaths(f"{self.root_dir}/NORMAL/") + absoluteFilePaths(f"{self.root_dir}/PNEUMONIA/")
        random.shuffle(self.imList)


    def __len__(self):
        return len(self.imList)


    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        label = 0 if 'NORMAL' in self.imList[idx] else 1
        img = pil.Image.open(self.imList[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        instance = {'image': self.transform(img), 'label': label}
        return instance


class ChestXRayImageDataView(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

        self.view = None

    def __len__(self):
        if self.view is None:
            return len(self.dataset)
        return len(self.view)

    def set_view(self, idx):
        self.view = idx

    def __getitem__(self, idx):
        if self.view is None:
            return self.dataset[idx]
        else:
            return self.dataset[self.view[idx]]

