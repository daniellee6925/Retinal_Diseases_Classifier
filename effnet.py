import torch
import torch.nn as nn
from math import ceil

base_model = [
    # expand ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]


phi_values = {
    # tuple of (phi_val, resolution, drop_rate)
    "b0": (0, 224, 0.2),  # alpha, beta, gamma, depth = alpha**gamma
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}


class CNNBlock(nn.Module):
    """
    A convolutional block consisting of a convolutional layer, batch normalization,
    and a SiLU activation function. Used for feature extraction in CNN-based architectures.

    Args:
        in_channels (int): Number of input channels in the image or feature map.
        out_channels (int): Number of output channels after applying the convolution.
        kernel_size (int): Size of the convolutional filter.
        stride (int): Step size for moving the filter across the input.
        padding (int): Amount of padding added to the input to maintain spatial dimensions.
        groups (int, optional): Number of groups for grouped convolution. Defaults to 1 (standard convolution).
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, groups=1
    ):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))


class SqueezeExcitation(nn.Module):
    """
    Implements a Squeeze-and-Excitation (SE) block that adaptively recalibrates channel-wise feature responses.
    It enhances important features while suppressing less useful ones.

    Args:
        in_channels (int): Number of input channels in the feature map.
        reduced_dim (int): Reduced dimension for the squeeze operation, controlling the compression ratio.
    """

    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (channel, height, width) -> (channel, 1, 1)
            nn.Conv2d(in_channels, reduced_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)  # how much the value in the channel is prioritized


class InvertedResidualBlock(nn.Module):
    """
    Implements an Inverted Residual Block with optional Squeeze-and-Excitation (SE)
    and Stochastic Depth for efficient deep learning.

    Args:
        in_channels (int): Number of input channels in the feature map.
        out_channels (int): Number of output channels after processing.
        kernel_size (int): Size of the depthwise convolution filter.
        stride (int): Step size for the convolution.
        padding (int): Amount of padding added to maintain spatial dimensions.
        expand_ratio (int): Factor by which the input channels are expanded in the first step.
        reduction (int, optional): Reduction ratio for the Squeeze-and-Excitation block. Defaults to 4.
        survival_prob (float, optional): Probability of keeping the layer active in Stochastic Depth. Defaults to 0.8.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        expand_ratio,
        reduction=4,  # for squeeze excitation,
        survival_prob=0.8,  # for stochastic
    ):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = 0.8
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = CNNBlock(
                in_channels, hidden_dim, kernel_size=3, stride=1, padding=1
            )
        self.conv = nn.Sequential(  # depth wise convolution
            CNNBlock(
                hidden_dim,
                hidden_dim,
                kernel_size,
                stride,
                padding,
                groups=hidden_dim,
            ),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def stochastic_depth(self, x):
        if not self.training:
            return x
        # 1 if less than survival prob, else 0
        binary_tensor = (
            torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        )

        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    """
    Implements the EfficientNet architecture, a family of scalable convolutional neural networks
    that balance accuracy and efficiency by scaling depth, width, and resolution.

    Args:
        version (str): Specifies the EfficientNet variant (e.g., "b0", "b1", "b2", etc.),
                       which determines scaling factors for width, depth, and dropout.
        num_classes (int): Number of output classes for classification.
    """

    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(last_channels, num_classes)
        )

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha**phi
        width_factor = beta**phi
        return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            # always divisble by 4
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layer_repeats = ceil(repeats * depth_factor)
            for layer in range(layer_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    )
                )
                in_channels = out_channels
        features.append(
            CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
        )

        return nn.Sequential(*features)

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))


"""Test for checking correct model implementation """


def test():
    device = "cuda"
    version = "b2"
    phi, res, drop_rate = phi_values[version]
    num_examples, num_classes = 4, 10
    x = torch.randn((num_examples, 3, res, res)).to(device)
    model = EfficientNet(version=version, num_classes=num_classes).to(device)

    print(model(x).shape)  # (num_examples, num_classes)
