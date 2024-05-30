import h5py
import torch
from torch.utils.data import Dataset

class H5Dataset(Dataset):
    def __init__(self, h5_file_path, device):
        self.h5_file_path = h5_file_path
        self.device = device

        # Save the keys - Sample_{i}
        with h5py.File(self.h5_file_path, 'r') as hf:
            self.keys = list(hf.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.h5_file_path, 'r') as hf:
            sample_name = self.keys[idx]
            data = torch.tensor(hf[sample_name]['data'][:]).to(self.device)
            label = torch.tensor(hf[sample_name]['label'][()]).to(self.device)
        
        return data, label
