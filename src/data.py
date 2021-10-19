import os

import pandas as pd
from PIL import Image

from torch.utils import data
from torchvision import transforms


HEIGHT = 224
WIDTH = 224


class Dataset(data.Dataset):
    def __init__(self, csv_path, transform):
        self.transform = transform
        # self.root = "../../data/recruit/food-101"
        self.root = os.path.join("data","food-101")
        data = pd.read_csv(os.path.join(self.root, csv_path))
        self.paths = data["path"]
        self.labels = data["label"]
        self.length = len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = self.transform(Image.open(os.path.join(self.root, path)))
        img = img.expand((3, HEIGHT, WIDTH))
        label = self.labels[idx]
        return path, img, label

    def __len__(self):
        return self.length


def get_train_loader(csv_path, batch_size, num_workers):
    """Returns data set and data loader for training."""
    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.ToTensor()])
    dataset = Dataset(csv_path, transform)
    loader = data.DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=True,
                             shuffle=True)
    return dataset, loader


def get_test_loader(csv_path, batch_size, num_workers):
    """Returns data set and data loader for evaluation."""
    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.ToTensor()])
    dataset = Dataset(csv_path, transform)
    loader = data.DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=True,
                             shuffle=False)
    return dataset, loader
