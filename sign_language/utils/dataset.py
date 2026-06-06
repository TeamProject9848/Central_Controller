import os
import numpy as np
import torch
from torch.utils.data import Dataset

def normalize_landmarks(sequence):
    """
    sequence: (T, 63)
    Normalization:
    - subtract wrist (landmark 0)
    - scale by palm size
    """
    seq = sequence.reshape(sequence.shape[0], 21, 3)

    wrist = seq[:, 0:1, :]          # (T,1,3)
    seq = seq - wrist               # translation invariance

    palm = seq[:, 9, :]              # middle finger MCP
    scale = np.linalg.norm(palm, axis=1, keepdims=True) + 1e-6
    seq = seq / scale[:, None, :]    # scale invariance

    return seq.reshape(sequence.shape[0], -1)

class AlphabetDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))

        for idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls)
            for file in os.listdir(cls_dir):
                if file.endswith(".npy"):
                    self.samples.append(os.path.join(cls_dir, file))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = np.load(self.samples[idx])       # (T, 63)
        x = normalize_landmarks(x)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.labels[idx])
        return x, y
