import argparse

from dataset import H5Dataset
from model_builder import ResNet18, ResNet50, VITBase, Mixer
from engine import train_model, test_model
from utils import save_model

import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader

def main(args, device):
    best_loss = float('inf')
    early_stopping_counter = 0
    patience = args.patience

    if args.model == 'resnet18':
        model = ResNet18(args.num_classes)
    elif args.model == 'resnet50':
        model = ResNet50(args.num_classes)
    elif args.model == 'vit_base':
        model = VITBase(args.num_classes)
    elif args.model == 'mixer':
        model = Mixer(args.num_classes)
    else:
        raise ValueError(f'Invalid model name: {args.model}')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(args.num_epochs):
        train_loss = train_model(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = test_model(model, test_loader, loss_fn, device)
        
        print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")

        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

        scheduler.step()
    
    save_model(model, '/models', args.model)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a model on a dataset')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset (without the .h5 extension)')
    parser.add_argument('--model', type=str, default='resnet18', help='Name of the model to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs to wait before early stopping if no improvement in validation loss')
    parser.add_argument('--num_classes', type=int, default=50, help='Number of classes')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64

    RANDOM_SEED = 1234
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    args = parse_arguments()
    
    dataset = H5Dataset(f'{args.dataset_name}.h5', device)
    validation_split = 0.7
    num_validation_samples = int(validation_split * len(dataset))
    num_training_samples = len(dataset) - num_validation_samples
    training_dataset, validation_dataset = random_split(dataset, [num_training_samples, num_validation_samples])
    train_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    main(args, device)