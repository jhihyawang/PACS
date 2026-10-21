import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention2D(nn.Module):
    """Position-aware attention that updates a target map from a context map."""

    def __init__(self, dim, heads=4):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.output_norm = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _position_encoding(height, width, dim, device, dtype):
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device, dtype=dtype),
            torch.linspace(-1, 1, width, device=device, dtype=dtype),
            indexing="ij",
        )
        coords = torch.stack([x, y], dim=-1).reshape(height * width, 2)
        pairs = (dim + 3) // 4
        freq = torch.exp(
            torch.arange(pairs, device=device, dtype=dtype)
            * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / max(pairs - 1, 1))
        )
        phase = coords.unsqueeze(-1) * freq
        encoding = torch.cat(
            [phase[..., 0].sin(), phase[..., 0].cos(), phase[..., 1].sin(), phase[..., 1].cos()],
            dim=-1,
        )
        if encoding.shape[-1] < dim:
            encoding = F.pad(encoding, (0, dim - encoding.shape[-1]))
        return encoding[..., :dim].unsqueeze(0)

    def forward(self, target, context):
        batch, channels, height, width = target.shape
        if context.shape[-2:] != (height, width):
            context = F.interpolate(context, size=(height, width), mode="bilinear", align_corners=False)
        pos = self._position_encoding(height, width, channels, target.device, target.dtype)
        target_tokens = target.flatten(2).transpose(1, 2)
        context_tokens = context.flatten(2).transpose(1, 2)
        q = self.query_norm(target_tokens + pos)
        kv = self.context_norm(context_tokens + pos)
        attended, _ = self.attention(q, kv, kv, need_weights=False)
        delta = self.output_norm(attended).transpose(1, 2).reshape(batch, channels, height, width)
        return self.alpha * delta


class SideInvariantCrossViewFusion(nn.Module):
    """Synchronous ipsilateral and optional same-view bilateral attention."""

    def __init__(self, dim, heads=4, mode="cross_view"):
        super().__init__()
        if mode not in {"ipsilateral", "cross_view"}:
            raise ValueError(f"Unsupported fusion mode: {mode}")
        self.mode = mode
        # Symmetric pairings share weights; each direction has its own zero gate.
        self.ipsi_cc_to_mlo = CrossAttention2D(dim, heads)
        self.ipsi_mlo_to_cc = CrossAttention2D(dim, heads)
        self.cc_l_to_r = CrossAttention2D(dim, heads)
        self.cc_r_to_l = CrossAttention2D(dim, heads)
        self.mlo_l_to_r = CrossAttention2D(dim, heads)
        self.mlo_r_to_l = CrossAttention2D(dim, heads)
        if mode == "cross_view":
            # Attention weights are shared for symmetric pairs; residual gates are not.
            self.edge_alpha = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(()))
                for name in (
                    "LCC_from_LMLO", "RCC_from_RMLO",
                    "LMLO_from_LCC", "RMLO_from_RCC",
                    "LCC_from_RCC", "RCC_from_LCC",
                    "LMLO_from_RMLO", "RMLO_from_LMLO",
                )
            })
            for module in (
                self.ipsi_cc_to_mlo, self.ipsi_mlo_to_cc,
                self.cc_l_to_r, self.cc_r_to_l,
                self.mlo_l_to_r, self.mlo_r_to_l,
            ):
                with torch.no_grad():
                    module.alpha.fill_(1.0)
                module.alpha.requires_grad_(False)

    def forward(self, feats):
        source = feats
        # Every update reads the original maps, not another edge's updated result.
        result = {
            "L-CC": source["L-CC"] + self.ipsi_cc_to_mlo(source["L-CC"], source["L-MLO"]),
            "R-CC": source["R-CC"] + self.ipsi_cc_to_mlo(source["R-CC"], source["R-MLO"]),
            "L-MLO": source["L-MLO"] + self.ipsi_mlo_to_cc(source["L-MLO"], source["L-CC"]),
            "R-MLO": source["R-MLO"] + self.ipsi_mlo_to_cc(source["R-MLO"], source["R-CC"]),
        }
        if self.mode == "cross_view":
            result["L-CC"] = source["L-CC"] + self.edge_alpha["LCC_from_LMLO"] * self.ipsi_cc_to_mlo(source["L-CC"], source["L-MLO"])
            result["R-CC"] = source["R-CC"] + self.edge_alpha["RCC_from_RMLO"] * self.ipsi_cc_to_mlo(source["R-CC"], source["R-MLO"])
            result["L-MLO"] = source["L-MLO"] + self.edge_alpha["LMLO_from_LCC"] * self.ipsi_mlo_to_cc(source["L-MLO"], source["L-CC"])
            result["R-MLO"] = source["R-MLO"] + self.edge_alpha["RMLO_from_RCC"] * self.ipsi_mlo_to_cc(source["R-MLO"], source["R-CC"])
            result["L-CC"] = result["L-CC"] + self.edge_alpha["LCC_from_RCC"] * self.cc_l_to_r(source["L-CC"], source["R-CC"])
            result["R-CC"] = result["R-CC"] + self.edge_alpha["RCC_from_LCC"] * self.cc_r_to_l(source["R-CC"], source["L-CC"])
            result["L-MLO"] = result["L-MLO"] + self.edge_alpha["LMLO_from_RMLO"] * self.mlo_l_to_r(source["L-MLO"], source["R-MLO"])
            result["R-MLO"] = result["R-MLO"] + self.edge_alpha["RMLO_from_LMLO"] * self.mlo_r_to_l(source["R-MLO"], source["L-MLO"])
        return result
