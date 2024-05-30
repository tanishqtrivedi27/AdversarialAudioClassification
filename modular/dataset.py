import h5py
import torch
from torch.utils.data import TensorDataset

def create_dataset(h5_file_path, device):
    data_list = []
    label_list = []

    with h5py.File(h5_file_path, 'r') as hf:
        for sample_name in hf.keys():
            data = torch.tensor(hf[sample_name]['data'][:])
            label = hf[sample_name]['label'][()]

            data_list.append(data)
            label_list.append(label)

    data_tensor = torch.stack(data_list).to(device)
    label_tensor = torch.tensor(label_list, dtype=torch.long).to(device)

    # Create a TensorDataset
    dataset = TensorDataset(data_tensor, label_tensor)
    return dataset