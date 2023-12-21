import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, X: list[list[int]], y: list[list[int]]):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int | torch.Tensor):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.X[idx]
        y = self.y[idx]

        return torch.LongTensor(x), torch.LongTensor(y)