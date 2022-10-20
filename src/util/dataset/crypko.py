import glob
import os

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class CrypkoDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        im = torchvision.io.read_image(fname)
        im = self.transform(im)
        return im

    def __len__(self):
        return len(self.fnames)


def crypko_dataloader(path, batch_size, n_jobs=0, shuffle=True):
    filenames = glob.glob(os.path.join(path, "*.jpg"))
    compose = [
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    dataset = CrypkoDataset(filenames, transform=transforms.Compose(compose))
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=False, num_workers=n_jobs, pin_memory=True)
    return dataloader
