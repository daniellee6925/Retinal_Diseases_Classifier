import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

"Hyperparameters"
img_size = 224
in_channels = 3
patch_size = 16
num_transformer_layers = 12
embedding_dim = 768
mlp_size = 3072
num_heads = 12
attn_dropout = 0.0
mlp_dropout = 0.1
embedding_dropout = 0.1
num_classes = (1000,)


# create patch embedding module
class PatchEmbeddings(nn.Module):
    """
    Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """

    def __init__(self, in_channels=3, patch_size=16, embedding_dim=768):
        super().__init__()
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % patch_size == 0, (
            "input image size must be divisible by patch size"
        )

        # forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)

        return x_flattened.permute(0, 2, 1)


class MultiHeadSelfAttentionBlock(nn.Module):
    """
    Multi-head Attention Block from nn.MultiheadAttention
    Core of the Transformer Architecture

    Args:
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
        num_heads (int): number of heads for multi-head. Defaults to 12.
        attn_dropout (float): Percent of layers to randomly "turn off" Defaults to 0.0.
    """

    def __init__(self, embedding_dim=768, num_heads=12, attn_dropout=0.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(
            query=x, key=x, value=x, need_weights=False
        )
        return attn_output


class MLPBlock(nn.Module):
    """
    Multi-Layer Perceptron Block
    Applies Non-Linear Transformation to "Learn" through back probagation

    Args:
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
        mlp_size (int): Number of hidden units in the MLP block, typically larger than embedding_dim. Defaults to 3072.
        dropout (float): Percent of layers to randomly "turn off" Defaults to 0.0.
    """

    def __init__(self, embedding_dim=768, mlp_size=3072, dropout=0.0, device="cuda"):
        super().__init__()
        self.device = device  # Store device

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim).to(self.device)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(p=dropout),
        ).to(self.device)

    def forward(self, x):
        x = x.to(self.device)  # Ensure input is on the same device
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block
    A single encoder block for a Vision Transformer (ViT), consisting of multi-head self-attention and an MLP block.

    Args:
        embedding_dim (int): Dimensionality of token embeddings. Defaults to 768.
        num_heads (int): Number of attention heads in the multi-head self-attention mechanism. Defaults to 12.
        mlp_size (int): Number of hidden units in the MLP block, typically larger than embedding_dim. Defaults to 3072.
        mlp_dropout (float): Dropout rate applied within the MLP block. Defaults to 0.1.
        attn_dropout (float): Dropout rate applied to the attention scores. Defaults to 0.0.
        device (str): Device on which the model runs ("cpu" or "cuda"). Defaults to "cuda".
    """

    def __init__(
        self,
        embedding_dim=768,
        num_heads=12,
        mlp_size=3072,
        mlp_dropout=0.1,
        attn_dropout=0.0,
        device="cuda",
    ):
        super().__init__()
        self.device = device  # Store device

        # Multi-head self-attention block
        self.msa_block = MultiHeadSelfAttentionBlock(
            embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout
        ).to(self.device)

        # MLP block
        self.mlp_block = MLPBlock(
            embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout
        ).to(self.device)

        # Layer Normalization (added before residual connections)
        self.layer_norm1 = nn.LayerNorm(embedding_dim).to(self.device)
        self.layer_norm2 = nn.LayerNorm(embedding_dim).to(self.device)

    def forward(self, x):
        x = x.to(self.device)  # Ensure input is on the same device

        # Apply LayerNorm before residual connections (standard in ViTs)
        x = x + self.msa_block(self.layer_norm1(x))
        x = x + self.mlp_block(self.layer_norm2(x))

        return x


class ViT(nn.Module):
    """
    Vision Transformer (ViT) Model
    Implements a Vision Transformer for image classification by dividing an image into patches and processing them through
    transformer encoder layers.

    Args:
        img_size (int): Size of the input image (assumed square). Defaults to 224.
        in_channels (int): Number of input image channels. Defaults to 3 (RGB).
        patch_size (int): Size of each image patch. Defaults to 16.
        num_transformer_layers (int): Number of transformer encoder layers. Defaults to 12.
        embedding_dim (int): Dimensionality of token embeddings. Defaults to 768.
        mlp_size (int): Number of hidden units in the MLP block, typically larger than embedding_dim. Defaults to 3072.
        num_heads (int): Number of attention heads in the multi-head self-attention mechanism. Defaults to 12.
        attn_dropout (float): Dropout rate applied to attention scores. Defaults to 0.0.
        mlp_dropout (float): Dropout rate applied within the MLP block. Defaults to 0.1.
        embedding_dropout (float): Dropout rate applied to the token embeddings. Defaults to 0.1.
        num_classes (int): Number of output classes for classification. Defaults to 1000 (ImageNet).
        device (str): Device on which the model runs ("cpu" or "cuda"). Defaults to "cuda".
    """

    def __init__(
        self,
        img_size=224,
        in_channels=3,
        patch_size=16,
        num_transformer_layers=12,
        embedding_dim=768,
        mlp_size=3072,
        num_heads=12,
        attn_dropout=0.0,
        mlp_dropout=0.1,
        embedding_dropout=0.1,
        num_classes=1000,
        device="cuda",
    ):
        super().__init__()

        self.device = device  # Store device

        assert img_size % patch_size == 0

        # Calculate number of patches
        self.num_patches = (img_size * img_size) // patch_size**2

        # Create learnable class embedding
        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, embedding_dim, device=self.device), requires_grad=True
        )

        # Create learnable position embedding
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embedding_dim, device=self.device),
            requires_grad=True,
        )

        # Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # Create patch embedding layer
        self.patch_embedding = PatchEmbeddings(
            in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim
        ).to(self.device)

        # Create transformer encoder blocks
        self.transformer_encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_size=mlp_size,
                    mlp_dropout=mlp_dropout,
                ).to(self.device)
                for _ in range(num_transformer_layers)
            ]
        )

        # Create classifier head
        self.classifer = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes),
        ).to(self.device)

    def forward(self, x):
        x = x.to(self.device)  # Ensure input is on the same device
        batch_size = x.shape[0]

        # Create class token embeddings for each batch
        class_tokens = self.class_embedding.expand(batch_size, -1, -1)

        # Create patch embedding
        x = self.patch_embedding(x)

        # Concatenate class tokens
        x = torch.cat((class_tokens, x), dim=1)

        # Add positional embeddings
        x = self.position_embedding + x

        # Run embedding dropout
        x = self.embedding_dropout(x)

        # Run through transformer encoder layers
        x = self.transformer_encoder(x)

        # Pass through classifier
        x = self.classifer(x[:, 0])  # Extract class token

        return x
