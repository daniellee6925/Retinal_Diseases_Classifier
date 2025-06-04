"""This program contains functions for training and testing a PyTorch Model"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision
import effnet
import ViT
from tqdm.auto import tqdm  # proccess bar
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    binary: bool = False,
) -> Tuple[float, float]:
    """Trains a PyTorch Model for one Epoch
    1. Train Mode
    2. Send target to device
    3. Forward Pass
    4. Loss Calculation
    5. Gradients sent to zero
    6. backward pass
    7. optimizer step

    Args:
    model: PyTorch Model
    dataloader: DataLoader instance to be trained on
    loss_fn: PyTorch loss function to minimize
    optimzer: PyTorch Optimizer
    device: target device to compute ("cpu", "cuda")
    binary: True if binary classfication

    Returns:
    A tuple of training loss, training accuracy, training precision, training recall, training f1-score
    (train_loss, train_accuracy, train_precision, train_recall, train_f1)
    """

    # put model on train mode
    model.eval()

    # set train loss and acc
    train_loss, train_acc = 0, 0
    train_precision, train_recall, train_f1 = 0, 0, 0

    # Loop through batches in DataLoader
    for batch, (X, y) in enumerate(dataloader):
        # send data to target device
        X, y = X.to(device), y.to(device)

        if binary:
            # for binary classification
            y = y.unsqueeze(1).to(torch.float32)

        # 1. Forward Pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metrics
        if binary:
            probs = torch.sigmoid(y_pred)  # Convert logits to probabilities
            y_pred_class = (
                probs >= 0.4
            ).int()  # Convert probabilities to binary labels

        else:
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)

        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        # Calculate and accumulate precision, recall, and F1-score (macro-averaged for multi-class classification)
        if binary:
            # Convert tensors to NumPy for sklearn metrics
            y_true_np = y.cpu().numpy()
            y_pred_np = y_pred_class.cpu().numpy()
            train_precision += precision_score(
                y_true_np, y_pred_np, pos_label=0, average="binary", zero_division=0
            )
            train_recall += recall_score(
                y_true_np, y_pred_np, pos_label=0, average="binary", zero_division=0
            )
            train_f1 += f1_score(
                y_true_np, y_pred_np, pos_label=0, average="binary", zero_division=0
            )

    # Adjust metrics to get average over per batch
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    train_precision /= len(dataloader)
    train_recall /= len(dataloader)
    train_f1 /= len(dataloader)

    # return output
    return train_loss, train_acc, train_precision, train_recall, train_f1


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    binary: bool = False,
) -> Tuple[float, float]:
    """Test a PyTorch Model for one Epoch
    1. Eval Mode
    2. Send target to device
    3. Forward Pass
    4. Loss Calculation

    Args:
    model: PyTorch Model
    dataloader: DataLoader instance to be trained on
    loss_fn: PyTorch loss function to minimize
    optimzer: PyTorch Optimizer
    device: target device to compute ("cpu", "cuda")
    binary: True if binary classfication

    Returns:
    A tuple of test loss, test accuracy, test precision, traintesting recall, test f1-score
    (test_loss, test_accuracy, test_precision, test_recall, test_f1)
    """
    # put model on evaluation mode
    model.eval()

    # setup test loss and test acc
    test_loss, test_acc = 0, 0
    test_precision, test_recall, test_f1 = 0, 0, 0

    # turn on inference mode on torch
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            # send data to target device
            X, y = X.to(device), y.to(device)

            if binary:
                # for binary classification
                y = y.unsqueeze(1).to(torch.float32)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. calculate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # 3. caclulate accuracy
            if binary:
                probs = torch.sigmoid(
                    test_pred_logits
                )  # Convert logits to probabilities
                test_pred_labels = (
                    probs >= 0.6
                ).int()  # Convert probabilities to binary labels
            else:
                test_pred_labels = torch.argmax(
                    torch.softmax(test_pred_logits, dim=1), dim=1
                )
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

            # Calculate and accumulate precision, recall, and F1-score (macro-averaged for multi-class classification)
            if binary:
                # Convert tensors to NumPy for sklearn metrics
                y_true_np = y.cpu().numpy()
                y_pred_np = test_pred_labels.cpu().numpy()
                test_precision += precision_score(
                    y_true_np, y_pred_np, pos_label=0, average="binary", zero_division=0
                )
                test_recall += recall_score(
                    y_true_np, y_pred_np, pos_label=0, average="binary", zero_division=0
                )
                test_f1 += f1_score(
                    y_true_np, y_pred_np, pos_label=0, average="binary", zero_division=0
                )

    # adjust metrics to get average per batch
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    test_precision /= len(dataloader)
    test_recall /= len(dataloader)
    test_f1 /= len(dataloader)

    # return outputs
    return test_loss, test_acc, test_precision, test_recall, test_f1


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    writer: torch.utils.tensorboard.writer.SummaryWriter = None,
    binary: bool = False,
) -> Dict[str, list[float]]:
    """
    Train and Tests a PyTorch Model
    -Passes target model through train and test steps for a number of epochs
    -Calculate, print, and sotre evaluation metrics

    Args:
    model: PyTorch Model
    train_dataloader: DataLoader instance to be trained on
    test_dataloader: DataLoader instance to be tested on
    optimzer: PyTorch Optimizer
    loss_fn: PyTorch loss function to minimize
    epochs: integer indicating how many epochs to train on
    device: target device to compute ("cpu", "cuda")
    writer: A SummaryWriter() instance to log model results to
    binary: True if binary classfication

    Returns:
    Dictionary of training and testing loss, accuracy, precision, recall, f1-score
    {train_loss: [...], train_acc:[...], test_loss[...], test_acc[...], train_pre[...]..., test_f1[...]}
    """

    # create results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "train_prec": [],
        "test_prec": [],
        "train_rec": [],
        "test_rec": [],
        "train_f1": [],
        "test_f1": [],
    }

    # loop through training and test loos for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            binary=binary,
        )
        test_loss, test_acc, test_prec, test_rec, test_f1 = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            binary=binary,
        )
        # print stats
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss : {train_loss:.4f} | "
            f"train_acc : {train_acc:.4f} | "
            f"train_prec : {train_prec:.4f} | "
            f"train_rec : {train_rec:.4f} | "
            f"train_f1 : {train_f1:.4f} | "
            f"test_loss : {test_loss:.4f} | "
            f"test_acc : {test_acc:.4f} | "
            f"test_prec : {test_prec:.4f} | "
            f"test_rec : {test_rec:.4f} | "
            f"test_f1 : {test_f1:.4f} | "
        )
        # update results dict
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_prec"].append(train_prec)
        results["train_rec"].append(train_rec)
        results["train_f1"].append(train_f1)

        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["test_prec"].append(test_prec)
        results["test_rec"].append(test_rec)
        results["test_f1"].append(test_f1)

        # See if there's a writer, if so, log to it
        if writer:
            # Add results to SummaryWriter
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss},
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="Accuracy",
                tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc},
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="Precision",
                tag_scalar_dict={"train_prec": train_prec, "test_prec": test_prec},
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="Recall",
                tag_scalar_dict={"train_rec": train_rec, "test_rec": test_rec},
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="f1-Score",
                tag_scalar_dict={"train_f1": train_f1, "test_f1": test_f1},
                global_step=epoch,
            )

            # Close the writer
            writer.close()
        else:
            pass

    # return results
    return results


def create_writer(
    experiment_name: str, model_name: str, extra: str = None
) -> torch.utils.tensorboard.writer.SummaryWriter():
    """
    Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
    experiment_name (str): Name of experiment.
    model_name (str): Name of model.
    extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """

    from datetime import datetime
    import os

    # get timestamp of current date
    timestamp = datetime.now().strftime(
        "%Y-%m-%d"
    )  # returns current date in YYYY-MM-DD format

    if extra:
        # create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    print(f"[INFO] Created SummaryWriter and saved to: {log_dir}")

    return SummaryWriter(log_dir=log_dir)


def create_pretrained_effnet_model(
    version: str = "b2", num_classes: int = 2, seed: int = 42
):
    """Creates a pretrained EfficientNet feature extractor model for a specified version.

    Args:
        version (str, optional): EfficientNet version (e.g., "b0", "b1", ..., "b7"). Defaults to "b2".
        num_classes (int, optional): Number of classes in the classifier head. Defaults to 2.
        seed (int, optional): Random seed value for reproducibility. Defaults to 42.

    Returns:
        model (torch.nn.Module): EfficientNet feature extractor model.
        transforms (torchvision.transforms): Image transforms for the specified EfficientNet version.
    """
    # Validate version input
    valid_versions = [f"b{i}" for i in range(8)]  # Supports b0 to b7
    if version not in valid_versions:
        raise ValueError(
            f"Invalid EfficientNet version '{version}'. Choose from {valid_versions}."
        )

    # Dynamically get the correct weights and model
    effnet_class = getattr(torchvision.models, f"efficientnet_{version}")
    weights_class = getattr(
        torchvision.models, f"EfficientNet_{version.upper()}_Weights"
    )

    weights = weights_class.DEFAULT
    transforms = weights.transforms()
    model = effnet_class(weights=weights)

    # Freeze all layers in the base model
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the last few blocks in EfficientNet
    for name, param in model.named_parameters():
        if (
            # "features.6" in name or
            "features.7" in name
        ):  # Adjust for different EfficientNet versions
            param.requires_grad = True  # Enable training for deeper layers

    # Set a random seed for reproducibility before modifying the classifier
    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(
            in_features=model.classifier[1].in_features, out_features=num_classes
        ),
    )
    return model, transforms


def create_untrained_effnet_model(
    version: str = "b2", num_classes: int = 2, seed: int = 42
):
    """Creates an EfficientNetB2 feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of classes in the classifier head. Defaults to 2.
        seed (int, optional): random seed value. Defaults to 42.

    Returns:
        model (torch.nn.Module): EffNetB2 feature extractor model.
        transforms (torchvision.transforms): EffNetB2 image transforms.
    """
    # 1, 2, 3. Create EffNetB2 pretrained weights, transforms and model
    model = effnet.EfficientNet(version=version, num_classes=num_classes)
    weights_class = getattr(
        torchvision.models, f"EfficientNet_{version.upper()}_Weights"
    )
    weights = weights_class.DEFAULT
    transforms = weights.transforms()

    return model, transforms


def create_untrained_ViT_model(
    version: str = "B_16", num_classes: int = 2, seed: int = 42
):
    """Creates an Vision Transfomers feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of classes in the classifier head. Defaults to 2.
        seed (int, optional): random seed value. Defaults to 42.

    Returns:
        model (torch.nn.Module): ViT feature extractor model.
        transforms (torchvision.transforms): EffNetB2 image transforms.
    """
    # 1, 2, 3. Create EffNetB2 pretrained weights, transforms and model
    model = ViT.ViT(version=version, num_classes=num_classes)
    weights_class = getattr(torchvision.models, f"ViT_{version}_Weights")
    weights = weights_class.DEFAULT
    transforms = weights.transforms()

    return model, transforms
