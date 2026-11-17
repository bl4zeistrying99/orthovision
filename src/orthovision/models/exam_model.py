"""Exam-level model: shared backbone per slice -> masked pool over slices (per plane)
-> concat 3 planes -> dropout -> linear head with one logit per label.

Implements synopsis 4.3 (slice -> series -> exam) and 4.5 (3 binary heads, one encoder).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def masked_pool(feats: torch.Tensor, mask: torch.Tensor, mode: str = "max") -> torch.Tensor:
    """feats (B,S,F), mask (B,S) bool (True = real slice) -> (B,F). Padding never wins."""
    feats = feats.float()
    if mode == "max":
        return feats.masked_fill(~mask.unsqueeze(-1), torch.finfo(feats.dtype).min).max(1).values
    if mode == "mean":
        m = mask.unsqueeze(-1).to(feats.dtype)
        return (feats * m).sum(1) / m.sum(1).clamp(min=1)
    raise ValueError(f"unknown pooling '{mode}'")


class ExamModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_features: int,
        planes: list[str],
        n_labels: int = 3,
        dropout: float = 0.3,
        pooling: str = "max",
        slice_chunk: int = 24,
        grad_checkpoint: bool = False,
        freeze_bn: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.planes = list(planes)
        self.pooling = pooling
        self.slice_chunk = slice_chunk
        self.grad_checkpoint = grad_checkpoint
        self.freeze_bn = freeze_bn
        self.dropout = nn.Dropout(dropout)      # the layer MC Dropout will keep active (Phase 4)
        self.head = nn.Linear(num_features * len(self.planes), n_labels)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """(N,3,H,W) -> (N,F), in chunks to bound memory."""
        outs = []
        for chunk in x.split(self.slice_chunk):
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                outs.append(checkpoint(self.backbone, chunk, use_reentrant=False))
            else:
                outs.append(self.backbone(chunk))
        return torch.cat(outs, dim=0)

    def forward(self, series: dict[str, torch.Tensor], mask: dict[str, torch.Tensor]):
        pooled = []
        for p in self.planes:
            x = series[p]                                   # (B,S,3,H,W)
            b, s = x.shape[:2]
            f = self._encode(x.flatten(0, 1)).reshape(b, s, -1)
            pooled.append(masked_pool(f, mask[p], self.pooling))   # (B,F)
        return self.head(self.dropout(torch.cat(pooled, dim=1)))   # logits (B, n_labels)


def build_model(name: str, cfg: dict, pretrained: bool = True) -> ExamModel:
    import timm

    backbone = timm.create_model(cfg["models"]["timm_ids"][name], pretrained=pretrained,
                                 num_classes=0)
    t = cfg["train"]
    return ExamModel(
        backbone, backbone.num_features, cfg["data"]["planes"], len(cfg["data"]["labels"]),
        cfg["models"]["dropout"], t["pooling"], t["slice_chunk"], t["grad_checkpoint"],
        t["freeze_bn"],
    )
