import os
import argparse

import pandas as pd
import h5py

import torch
import torchvision.transforms
import torchaudio

"""
python data_setup.py --annotations_file /data/ESC-50-master/ESC-50-master/meta/esc50.csv --audio-dir /data/ESC-50-master/ESC-50-master/audio --sample_rate 44100
"""

def create_h5_dataset(annotations_file, audio_dir, sample_rate, device, h5_file_path):
    transform = torchvision.transforms.Compose([
        torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=2048, hop_length=512).to(device),
        torchaudio.transforms.AmplitudeToDB().to(device),
        torchvision.transforms.Resize((224, 224), antialias=True).to(device),
        torchvision.transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))
    ])

    annotations = pd.read_csv(annotations_file)
    
    with h5py.File(h5_file_path, 'w') as hf:
        for i, row in annotations.iterrows():
            audio_sample_path = os.path.join(audio_dir, row['filename'])
            label = row['label']
            signal, sr = torchaudio.load(audio_sample_path)
            signal = signal.to(device)

            if signal.shape[0] > 1:
                signal = torch.mean(signal, dim=0, keepdim=True)

            # resampling to make the sample rate as sample_rate
            if (sr != sample_rate):
                resample_transform = torchaudio.transforms.Resample(sr, sample_rate).to(device)
                signal = resample_transform(signal)

            # cut if more samples than sample_rate * duration
            num_samples = sample_rate * 5
            if (signal.shape[1] > num_samples):
                signal = signal[:, : num_samples]

            # right pad 0s if less than num_samples
            if (signal.shape[1] < num_samples):
                num_missing = num_samples - signal.shape[1]
                last_dim_padding = (0, num_missing)
                signal = torch.nn.functional.pad(signal, last_dim_padding)

            mel_spectrogram = transform(signal)
            
            grp = hf.create_group(f'sample_{i}')
            grp.create_dataset('data', data=mel_spectrogram.cpu().numpy())
            grp.create_dataset('label', data=label)

    print("Dataset saved to", h5_file_path)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Create HDF5 dataset from dataset folder')
    parser.add_argument('--annotations_file', type=str, required=True, help='Path to the annotations CSV file')
    parser.add_argument('--audio_dir', type=str, required=True, help='Directory containing audio files')
    parser.add_argument('--sample_rate', type=int, default=44100, help='Sample rate of the audio files')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    target_dir = '/data'
    os.makedirs(target_dir, exist_ok=True)
    h5_file_path = os.path.join(target_dir, f'{args.dataset_name}.h5')
    create_h5_dataset(args.annotations_file, args.audio_dir, args.sample_rate, device, h5_file_path)
