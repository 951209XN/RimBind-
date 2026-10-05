"""Single end-to-end structure-conditioned ESM3 / anchored-BiMamba model.

Variants A--F differ only by the pre-registered stage additions.  ESM3 values
are passed in as precomputed frozen tensors; this module has no ESM3 backbone
and consequently cannot fine-tune it.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from mamba_ssm import Mamba


VARIANTS = ("A", "B", "C", "D", "E", "F")


def knn_indices(pos: torch.Tensor, k: int) -> torch.Tensor:
    """Return KNN indices per residue, excluding self when possible."""
    n = len(pos)
    if n == 1:
        return torch.zeros(1, 1, dtype=torch.long, device=pos.device)
    dist = torch.cdist(pos.float(), pos.float())
    dist.fill_diagonal_(float("inf"))
    return dist.topk(min(k, n - 1), largest=False).indices


class SinusoidalPosition(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        p = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(p * scale)
        pe[:, 1::2] = torch.cos(p * scale[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x) > len(self.pe):
            raise ValueError(f"sequence length {len(x)} exceeds {len(self.pe)}")
        return x + self.pe[: len(x)].to(x)


class EGNNLayer(nn.Module):
    """Residue graph layer using only relative C-alpha distance."""
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        edge_dim = dim // 2
        self.edge = nn.Sequential(nn.Linear(2 * dim + 1, edge_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(edge_dim, edge_dim), nn.SiLU())
        self.node = nn.Sequential(nn.Linear(dim + edge_dim, 2 * dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, pos: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        n, k = idx.shape
        src = torch.arange(n, device=h.device).unsqueeze(1).expand(n, k).reshape(-1)
        dst = idx.reshape(-1)
        radial = (pos[src] - pos[dst]).square().sum(-1, keepdim=True).sqrt().clamp(max=30.0) / 10.0
        msg = self.edge(torch.cat([h[src], h[dst], radial], dim=-1)).reshape(n, k, -1).mean(1)
        return self.norm(h + self.node(torch.cat([h, msg], dim=-1)))


class P1Core(nn.Module):
    """Stage 1: SaProt sequence stream plus C-alpha KNN graph stream."""
    def __init__(self, saprot_dim: int, dim: int, layers: int, heads: int, k: int, dropout: float):
        super().__init__()
        self.k = k
        self.proj = nn.Sequential(nn.Linear(saprot_dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout))
        self.pos = SinusoidalPosition(dim)
        enc = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, "gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, layers)
        self.egnn = nn.ModuleList([EGNNLayer(dim, dropout) for _ in range(layers)])
        self.g_to_s = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.s_to_g = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(4 * dim, dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(dim))

    def sequence_only(self, saprot: torch.Tensor) -> torch.Tensor:
        h = self.proj(saprot.float())
        return self.transformer(self.pos(h).unsqueeze(0)).squeeze(0)

    def forward(self, saprot: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequence_only(saprot)
        idx = knn_indices(pos - pos.mean(0, keepdim=True), self.k)
        graph = self.proj(saprot.float())
        centered = pos - pos.mean(0, keepdim=True)
        for layer in self.egnn:
            graph = layer(graph, centered, idx)
        gs = self.g_to_s(graph.unsqueeze(0), seq.unsqueeze(0), seq.unsqueeze(0), need_weights=False)[0].squeeze(0)
        sg = self.s_to_g(seq.unsqueeze(0), graph.unsqueeze(0), graph.unsqueeze(0), need_weights=False)[0].squeeze(0)
        return self.fuse(torch.cat([seq, graph, gs, sg], dim=-1)), idx


class StructureConditionedESM3(nn.Module):
    """Stage 2: keep/self/struct competitive routing conditioned on P1 state."""
    def __init__(self, esm3_dim: int, dim: int, dropout: float):
        super().__init__()
        self.adapter = nn.Sequential(nn.LayerNorm(esm3_dim), nn.Linear(esm3_dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.simple_gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1))
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.router = nn.Sequential(nn.Linear(4 * dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 3))
        self.self_proj = nn.Linear(dim, dim, bias=False)
        self.struct_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def simple(self, u: torch.Tensor, esm3: torch.Tensor) -> torch.Tensor:
        e = self.adapter(esm3.float())
        gate = torch.sigmoid(self.simple_gate(torch.cat([u, e], dim=-1)))
        return self.norm(u + gate * e)

    def routed(self, u: torch.Tensor, esm3: torch.Tensor, knn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.adapter(esm3.float())
        neigh = e[knn]
        score = (self.q(u).unsqueeze(1) * self.k(neigh)).sum(-1) / math.sqrt(u.size(-1))
        attn = torch.softmax(score, dim=-1)
        c = (attn.unsqueeze(-1) * self.v(neigh)).sum(1)
        weights = torch.softmax(self.router(torch.cat([u, e, c, e - c], dim=-1)), dim=-1)
        # weights[:,0] is explicit keep/P1 option; it does not introduce a new feature.
        z = self.norm(u + weights[:, 1:2] * self.self_proj(e) + weights[:, 2:3] * self.struct_proj(c))
        return z, weights


class OrdinaryBiMamba(nn.Module):
    """Stage E control: bidirectional Mamba with simple concatenate/projection."""
    def __init__(self, dim: int, d_state: int, d_conv: int, expand: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.forward_mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.out = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.out_norm = nn.LayerNorm(dim)

    def states(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm(z).unsqueeze(0)
        f = self.forward_mamba(h).squeeze(0)
        b = torch.flip(self.backward_mamba(torch.flip(h, dims=[1])), dims=[1]).squeeze(0)
        return f, b

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        f, b = self.states(z)
        return self.out_norm(z + self.out(torch.cat([f, b], dim=-1)))


class AnchoredBiMamba(OrdinaryBiMamba):
    """Stage F: directional selection and P1-local-evidence anchor."""
    def __init__(self, dim: int, d_state: int, d_conv: int, expand: int, dropout: float):
        super().__init__(dim, d_state, d_conv, expand, dropout)
        self.direction = nn.Sequential(nn.Linear(4 * dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1))
        self.anchor = nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1))
        self.context_proj = nn.Linear(dim, dim, bias=False)
        self.evidence_proj = nn.Linear(dim, dim, bias=False)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, z: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f, b = self.states(z)
        direction = torch.sigmoid(self.direction(torch.cat([f, b, (f - b).abs(), u], dim=-1)))
        context = direction * f + (1.0 - direction) * b
        anchor = torch.sigmoid(self.anchor(torch.cat([u, z, context], dim=-1)))
        h = self.final_norm(z + anchor * self.context_proj(context) + (1.0 - anchor) * self.evidence_proj(u))
        return h, direction, anchor


class SingleNABindModel(nn.Module):
    def __init__(self, variant: str, saprot_dim: int = 1280, esm3_dim: int = 1536, dim: int = 256, layers: int = 3, heads: int = 8, k: int = 16, dropout: float = 0.2, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant}; choices={VARIANTS}")
        self.variant = variant
        self.p1 = P1Core(saprot_dim, dim, layers, heads, k, dropout)
        self.esm = StructureConditionedESM3(esm3_dim, dim, dropout) if variant in {"C", "D", "E", "F"} else None
        self.mamba = OrdinaryBiMamba(dim, d_state, d_conv, expand, dropout) if variant == "E" else None
        self.anchored = AnchoredBiMamba(dim, d_state, d_conv, expand, dropout) if variant == "F" else None
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, 1))

    def forward(self, saprot_list: list[torch.Tensor], pos_list: list[torch.Tensor], esm3_list: list[torch.Tensor], return_aux: bool = False):
        if not (len(saprot_list) == len(pos_list) == len(esm3_list)):
            raise ValueError("one SaProt/coordinate/ESM3 tensor is required per protein")
        outputs, aux_rows = [], []
        for saprot, pos, esm3 in zip(saprot_list, pos_list, esm3_list):
            if len(saprot) != len(pos) or len(saprot) != len(esm3):
                raise ValueError(f"residue-length mismatch: SaProt={len(saprot)}, pos={len(pos)}, ESM3={len(esm3)}")
            if self.variant == "A":
                h = self.p1.sequence_only(saprot)
                aux = {}
            else:
                u, knn = self.p1(saprot, pos)
                h, aux = u, {"knn": knn}
                if self.variant == "C":
                    h = self.esm.simple(u, esm3)
                elif self.variant in {"D", "E", "F"}:
                    h, weights = self.esm.routed(u, esm3, knn)
                    aux["router"] = weights
                if self.variant == "E":
                    h = self.mamba(h)
                elif self.variant == "F":
                    h, direction, anchor = self.anchored(h, u)
                    aux["direction"] = direction
                    aux["anchor"] = anchor
            outputs.append(self.head(h).squeeze(-1))
            aux_rows.append(aux)
        logits = torch.cat(outputs, dim=0)
        return (logits, aux_rows) if return_aux else logits
