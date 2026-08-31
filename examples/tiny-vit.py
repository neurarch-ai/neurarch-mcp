"""A small ViT-ish classifier, written the way people actually write one.

It is here so `npx -y neurarch-mcp examples/tiny-vit.py` has something to read
with no app, no export step and no API key. It also carries a real design bug
for `lint_model` to find, rather than being a clean file that proves nothing:
the attention block has an embed_dim that its head count does not divide.
"""
import torch.nn as nn


class TinyViT(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.patch = nn.Conv2d(3, 258, kernel_size=16, stride=16)
        self.norm1 = nn.LayerNorm(258)
        self.attn = nn.MultiheadAttention(embed_dim=258, num_heads=8)
        self.drop = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(258)
        self.mlp = nn.Linear(258, 1024)
        self.act = nn.GELU()
        self.proj = nn.Linear(1024, 258)
        self.head = nn.Linear(258, num_classes)

    def forward(self, x):
        x = self.patch(x)
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop(x)
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.act(x)
        x = self.proj(x)
        return self.head(x)
