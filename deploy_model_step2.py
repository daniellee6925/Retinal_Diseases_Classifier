import matplotlib.pyplot as plt
import torch
import torchvision


from torch import nn
from torchvision import transforms
from pathlib import Path

import engine
import helper_functions
import utils
import data_setup
import predict

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    # your main code here
    # Setup directory paths to train and test images
    train_dir = "data/Train"
    val_dir = "data/Val"
    test_dir = "data/Test"

    effnetb3, effnetb3_transforms = engine.create_pretrained_effnet_model(
        version="b3", num_classes=5, seed=42
    )

    train_dataloader_effnetb3, test_dataloader_effnetb3, class_names = (
        data_setup.create_dataloaders(
            train_dir=train_dir,
            test_dir=test_dir,
            transforms=effnetb3_transforms,
            batch_size=32,
        )
    )
    print(f"class Names: {class_names}")
    print(f"Total batches: {len(train_dataloader_effnetb3)}")
    print(f"Dataset size: {len(train_dataloader_effnetb3.dataset)}")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        params=effnetb3.parameters(), lr=1e-4, weight_decay=1e-5
    )

    # Setup loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    helper_functions.set_seeds()
    effnetb3.to(device)
    # Set seeds for reproducibility and train the model
    helper_functions.set_seeds()
    effnetb3_results = engine.train(
        model=effnetb3,
        train_dataloader=train_dataloader_effnetb3,
        test_dataloader=test_dataloader_effnetb3,
        epochs=10,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        binary=False,
    )

    import sys

    sys.exit()

    # Save the model
    utils.save_model(
        model=effnetb3,
        target_dir="models",
        model_name="pretrained_effnetb3_type_classification.pth",
    )

    # ----------------------------------------------

    # Get the model size in bytes then convert to megabytes
    pretrained_effnetb1_model_size = Path(
        "models/pretrained_effnetb0_disease_classification.pth"
    ).stat().st_size // (1024 * 1024)
    # Count number of parameters in EffNetB2
    effnetb1_total_params = sum(torch.numel(param) for param in effnetb3.parameters())

    # Create a dictionary with EffNetB2 statistics
    effnetb1_stats = {
        "test_loss": effnetb3_results["test_loss"][-1],
        "test_acc": effnetb3_results["test_acc"][-1],
        "number_of_parameters": effnetb1_total_params,
        "model_size (MB)": pretrained_effnetb1_model_size,
    }

    print(effnetb1_stats)

    # make predictions
    print(f"[INFO] Finding all filepaths ending with '.jpg' in directory: {test_dir}")
    test_data_paths = list(Path(test_dir).glob("*/*.png"))
    test_data_paths[:5]
    effnetb1_test_pred_dicts = predict.pred_and_store(
        paths=test_data_paths,
        model=effnetb3,
        transform=effnetb3_transforms,
        class_names=class_names,
        device="cpu",
    )  # make predictions on CPU

    print(effnetb1_test_pred_dicts[:2])
