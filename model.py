import torch
import torchvision

from torch import nn


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
