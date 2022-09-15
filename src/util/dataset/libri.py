from torch.utils.data import Dataset


class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = y
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is None:
            return self.data[idx]
        else:
            return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)
