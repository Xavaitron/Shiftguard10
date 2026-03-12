"""
Compact Convolutional Transformer (CCT) for 32x32 image classification.
Based on "Escaping the Big Data Paradigm with Compact Transformers" (Hassani et al.)
Designed to train from scratch on small datasets — perfect for ShiftGuard10.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class ConvTokenizer(nn.Module):
    """
    Convolutional tokenizer that replaces ViT's linear patch embedding.
    Provides translation equivariance and better inductive bias for small data.
    """
    def __init__(self, in_channels=3, embed_dim=256, n_conv_layers=2, kernel_size=3):
        super().__init__()
        layers = []
        in_ch = in_channels
        for i in range(n_conv_layers):
            out_ch = embed_dim if i == n_conv_layers - 1 else embed_dim // 2
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                          stride=1, padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            if i < n_conv_layers - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = out_ch
        # Final pooling to reduce spatial dims
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 3, 32, 32) -> (B, embed_dim, H', W')
        x = self.conv(x)
        # Flatten spatial dims -> (B, embed_dim, num_tokens) -> (B, num_tokens, embed_dim)
        return x.flatten(2).transpose(1, 2)


class SequencePooling(nn.Module):
    """
    Attention-weighted sequence pooling (replaces CLS token).
    Learns to attend to the most informative tokens.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: (B, seq_len, embed_dim)
        attn_weights = F.softmax(self.attention(x), dim=1)  # (B, seq_len, 1)
        return (x * attn_weights).sum(dim=1)  # (B, embed_dim)


class TransformerEncoderLayer(nn.Module):
    """Standard Transformer encoder layer with pre-norm and stochastic depth."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=2.0,
                 dropout=0.1, attn_dropout=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop_path_rate = drop_path

    def _drop_path(self, x):
        """Stochastic depth: randomly drop entire residual branch."""
        if not self.training or self.drop_path_rate == 0.0:
            return x
        keep_prob = 1 - self.drop_path_rate
        mask = torch.rand(x.shape[0], 1, 1, device=x.device) < keep_prob
        return x * mask / keep_prob

    def forward(self, x):
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self._drop_path(attn_out)
        # Pre-norm FFN
        x = x + self._drop_path(self.mlp(self.norm2(x)))
        return x


class CCT(nn.Module):
    """
    Compact Convolutional Transformer.

    Architecture: Conv Tokenizer → Positional Embedding → Transformer Encoder → Sequence Pooling → Head

    Default config (CCT-7/3x1):
      ~3.7M params, designed for 32x32 images trained from scratch.
    """
    def __init__(
        self,
        img_size=32,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        num_heads=4,
        num_layers=7,
        mlp_ratio=2.0,
        n_conv_layers=2,
        kernel_size=3,
        dropout=0.1,
        attn_dropout=0.1,
        stochastic_depth=0.1,
    ):
        super().__init__()

        self.tokenizer = ConvTokenizer(
            in_channels=in_channels,
            embed_dim=embed_dim,
            n_conv_layers=n_conv_layers,
            kernel_size=kernel_size,
        )

        # Compute number of tokens after conv tokenizer
        # With n_conv_layers=2: 32 -> /2 (maxpool) -> /2 (final maxpool) = 8x8 = 64 tokens
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            num_tokens = self.tokenizer(dummy).shape[1]

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_tokens, embed_dim) * 0.02
        )
        self.pos_dropout = nn.Dropout(dropout)

        # Stochastic depth schedule (linear increase)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]

        self.transformer = nn.Sequential(*[
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                drop_path=dpr[i],
            )
            for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.pool = SequencePooling(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        # x: (B, 3, 32, 32)
        tokens = self.tokenizer(x)             # (B, num_tokens, embed_dim)
        tokens = self.pos_dropout(tokens + self.pos_embedding)
        tokens = self.transformer(tokens)      # (B, num_tokens, embed_dim)
        tokens = self.norm(tokens)
        pooled = self.pool(tokens)             # (B, embed_dim)
        return self.head(pooled)               # (B, num_classes)


def cct_7_3x1(num_classes=10, **kwargs):
    """CCT-7/3x1: 7 transformer layers, 3x3 conv tokenizer, 1 conv stage."""
    defaults = dict(
        embed_dim=256, num_heads=4, num_layers=7,
        mlp_ratio=2.0, n_conv_layers=2, kernel_size=3,
        dropout=0.1, attn_dropout=0.1, stochastic_depth=0.1,
    )
    defaults.update(kwargs)
    return CCT(num_classes=num_classes, **defaults)
