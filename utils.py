"""Contains Various Utility functions for PyTorch Model Training and Saving"""

from pathlib import Path

import torch


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """
    Saves a PyTorch model to a target directory

    Args:
        model: PyTorch model to save
        traget_dir: directory for saving the model
        model_name: name of model
    """

    # create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # create model save path
    assert model_name.endswith(".pth"), "model name should end with '.pth'"
    model_save_path = target_dir_path / model_name

    # save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
