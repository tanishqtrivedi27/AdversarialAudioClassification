import os
import torch

def save_model(model, target_dir, model_name):
    os.mkdir(target_dir, exist_ok=True)
    model_save_path = os.path.join(target_dir, f'{model_name}.pth')

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)