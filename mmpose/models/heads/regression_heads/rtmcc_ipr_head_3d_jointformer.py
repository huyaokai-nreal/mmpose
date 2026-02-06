# coding: utf-8
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmpose.evaluation.functional import keypoint_pck_accuracy
from mmpose.models.utils.tta import flip_coordinates
from mmpose.registry import MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import ConfigType, OptConfigType, OptSampleList

from mmpose.models.heads.coord_cls_heads import RTMCCHead3D


def _cfg_get(cfg: dict, key: str, default):
    v = cfg.get(key, default)
    return default if v is None else v


def _grid_sample_2d(
    feat_map: Tensor,
    coords_xy: Tensor,
    padding_mode: str = "border",
    align_corners: bool = False,
) -> Tensor:
    """Sample 2D feature map at normalized coordinates in [0, 1].

    Args:
        feat_map: [B, C, H, W]
        coords_xy: [B, K, 2] or [B, K, M, 2] in [0, 1] (x, y).

    Returns:
        If coords_xy is [B, K, 2]    -> [B, K, C]
        If coords_xy is [B, K, M, 2] -> [B, K, M, C]
    """
    squeeze_m = coords_xy.dim() == 3
    if squeeze_m:
        coords_xy = coords_xy.unsqueeze(2)  # [B,K,1,2]

    coords = coords_xy.clamp(0.0, 1.0)
    grid = coords * 2.0 - 1.0  # [-1,1]
    B, K, M, _ = grid.shape
    grid = grid.view(B, K * M, 1, 2)

    sampled = F.grid_sample(
        feat_map,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    ).squeeze(-1)  # [B,C,K*M]

    C = feat_map.size(1)
    sampled = sampled.view(B, C, K, M).permute(0, 2, 3, 1).contiguous()  # [B,K,M,C]
    return sampled.squeeze(2) if squeeze_m else sampled


def _grid_sample_2d_multilevel(
    feat_maps: Sequence[Tensor],  # list of [B,C,H,W], requires same H,W across levels
    coords_xy: Tensor,  # [B,K,L,M,2] in [0,1]
    padding_mode: str = "border",
    align_corners: bool = False,
) -> Tensor:
    """Multi-level sampling with a single grid_sample.

    Returns:
        sampled: [B, K, L, M, C]
    """
    assert coords_xy.dim() == 5, "coords_xy must be [B,K,L,M,2]"
    B, K, L, M, _ = coords_xy.shape
    assert len(feat_maps) == L, f"feat_maps length {len(feat_maps)} != L {L}"

    H0, W0 = feat_maps[0].shape[-2:]
    for fm in feat_maps[1:]:
        if fm.shape[-2:] != (H0, W0):
            raise AssertionError("All feat_maps must share same H,W for multilevel sampling.")

    fm = torch.stack(feat_maps, dim=0).transpose(0, 1).contiguous()  # [B,L,C,H,W]
    fm = fm.view(B * L, fm.size(2), H0, W0)

    grid = coords_xy.clamp(0.0, 1.0) * 2.0 - 1.0
    grid = grid.permute(0, 2, 1, 3, 4).contiguous()  # [B,L,K,M,2]
    grid = grid.view(B * L, K * M, 1, 2)

    sampled = F.grid_sample(
        fm,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    ).squeeze(-1)  # [B*L,C,K*M]

    C = sampled.size(1)
    sampled = sampled.view(B, L, C, K, M).permute(0, 3, 1, 4, 2).contiguous()  # [B,K,L,M,C]
    return sampled


class LayerScale(nn.Module):
    """Per-channel residual scaling for stabilizing deep transformer stacks."""

    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class SimCCToKeypoint3DPlus(nn.Module):

    def __init__(
        self,
        feat_w: int = 256,
        feat_h: int = 256,
        feat_d: int = 256,
        map_type: str = "softmax",
        temperature: Union[float, Tuple[float, float, float]] = 1.0,
        learnable_temperature: bool = False,
        temperature_min: float = 0.05,
        temperature_max: float = 10.0,
        linspace_denominator: str = "L",  # 'L' | 'L-1'
    ) -> None:
        super().__init__()
        self.map_type = str(map_type).lower()
        assert self.map_type in {"softmax", "elu", "raw"}

        self.linspace_denominator = str(linspace_denominator).lower()
        assert self.linspace_denominator in {"l", "l-1"}

        def _denom(L: int) -> float:
            if self.linspace_denominator == "l-1":
                return float(max(1, L - 1))
            return float(L)

        # normalized in [0,1)
        self.register_buffer(
            "linspace_x",
            torch.arange(feat_w, dtype=torch.float32) / _denom(feat_w),
            persistent=False,
        )
        self.register_buffer(
            "linspace_y",
            torch.arange(feat_h, dtype=torch.float32) / _denom(feat_h),
            persistent=False,
        )
        self.register_buffer(
            "linspace_z",
            torch.arange(feat_d, dtype=torch.float32) / _denom(feat_d),
            persistent=False,
        )

        if isinstance(temperature, (int, float)):
            t = (float(temperature), float(temperature), float(temperature))
        else:
            assert len(temperature) == 3
            t = (float(temperature[0]), float(temperature[1]), float(temperature[2]))

        self.temperature_min = float(temperature_min)
        self.temperature_max = float(temperature_max)

        if learnable_temperature:
            self.log_tau = nn.Parameter(torch.log(torch.tensor(t, dtype=torch.float32)))
            self.register_buffer("tau", torch.tensor(t, dtype=torch.float32), persistent=False)
        else:
            self.log_tau = None
            self.register_buffer("tau", torch.tensor(t, dtype=torch.float32), persistent=False)

    def get_base_tau(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.log_tau is None:
            tau = self.tau.to(device=device, dtype=dtype)
        else:
            tau = self.log_tau.exp().to(device=device, dtype=dtype)
        return tau.clamp(min=self.temperature_min, max=self.temperature_max)

    @staticmethod
    def _elu_normalize(featmaps: Tensor) -> Tensor:
        heatmaps = F.elu(featmaps) + 1.0
        heatmaps = heatmaps / heatmaps.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return heatmaps

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        z: Tensor,
        tau_factor: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Args:
        x/y/z: logits [B, K, Lx/Ly/Lz]
        tau_factor: broadcastable to [B, K, 3], multiplicative.
        """
        tau = self.get_base_tau(device=x.device, dtype=x.dtype).view(1, 1, 3)
        if tau_factor is not None:
            tau = tau * tau_factor.to(device=x.device, dtype=x.dtype)
        tau = tau.clamp(min=self.temperature_min, max=self.temperature_max)

        if self.map_type == "softmax":
            px = F.softmax(x / tau[..., 0:1], dim=-1)
            py = F.softmax(y / tau[..., 1:2], dim=-1)
            pz = F.softmax(z / tau[..., 2:3], dim=-1)
        elif self.map_type == "elu":
            px = self._elu_normalize(x)
            py = self._elu_normalize(y)
            pz = self._elu_normalize(z)
        else:
            px, py, pz = x, y, z

        lin_x = self.linspace_x.to(device=x.device, dtype=x.dtype)
        lin_y = self.linspace_y.to(device=x.device, dtype=x.dtype)
        lin_z = self.linspace_z.to(device=x.device, dtype=x.dtype)
        pred_x = (px * lin_x).sum(dim=-1, keepdim=True)
        pred_y = (py * lin_y).sum(dim=-1, keepdim=True)
        pred_z = (pz * lin_z).sum(dim=-1, keepdim=True)
        return pred_x, pred_y, pred_z


class TemperatureAdapter(nn.Module):
    """Produce per-joint per-axis temperature factor."""

    def __init__(
        self,
        enable: bool = False,
        source: str = "sigma_head",  # 'sigma_head' | 'simcc_conf' | 'hybrid'
        conf_type: str = "entropy",  # 'entropy' | 'pmax'
        factor_range: Tuple[float, float] = (0.70, 1.60),
        axiswise: bool = True,
        detach: bool = False,
        sigma_mode: str = "sigmoid",  # 'sigmoid' | 'tanh'
        hybrid_alpha: float = 0.5,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.enable = bool(enable)
        self.source = str(source).lower()
        assert self.source in {"sigma_head", "simcc_conf", "hybrid"}

        self.conf_type = str(conf_type).lower()
        assert self.conf_type in {"entropy", "pmax"}

        fmin, fmax = float(factor_range[0]), float(factor_range[1])
        assert fmin > 0 and fmax > 0 and fmax >= fmin
        self.fmin = fmin
        self.fmax = fmax

        self.axiswise = bool(axiswise)
        self.detach = bool(detach)

        self.sigma_mode = str(sigma_mode).lower()
        assert self.sigma_mode in {"sigmoid", "tanh"}

        self.hybrid_alpha = float(hybrid_alpha)
        self.eps = float(eps)

    def _sigma_to_factor(self, sigma_logits: Tensor) -> Tensor:
        s = sigma_logits.detach() if self.detach else sigma_logits
        if self.sigma_mode == "sigmoid":
            u = torch.sigmoid(s)
            factor = self.fmin + (self.fmax - self.fmin) * u
        else:
            u = torch.tanh(s)
            factor = 0.5 * (self.fmin + self.fmax) + 0.5 * (self.fmax - self.fmin) * u

        if not self.axiswise:
            factor = factor.mean(dim=-1, keepdim=True).expand_as(factor)
        return factor

    def _conf_to_factor_1d(self, logits: Tensor, tau: Tensor) -> Tensor:
        """logits [B,K,L] -> factor [B,K]; tau is Tensor (keeps grad)."""
        L = logits.size(-1)

        if self.conf_type == "pmax":
            p = F.softmax(logits / tau, dim=-1)
            if self.detach:
                p = p.detach()
            conf = p.max(dim=-1).values
            conf_min = 1.0 / max(1, L)
            denom = max(self.eps, 1.0 - conf_min)
            conf_norm = ((conf - conf_min) / denom).clamp(0.0, 1.0)
            return self.fmax - (self.fmax - self.fmin) * conf_norm

        # entropy via log_softmax 
        logp = F.log_softmax(logits / tau, dim=-1)
        p = logp.exp()
        if self.detach:
            p = p.detach()
            logp = logp.detach()

        ent = -(p * logp).sum(dim=-1)  # [B,K]
        ent_denom = max(self.eps, math.log(max(2, L)))
        ent_norm = (ent / ent_denom).clamp(0.0, 1.0)
        return self.fmin + (self.fmax - self.fmin) * ent_norm

    def _simcc_to_factor(self, feat_x: Tensor, feat_y: Tensor, feat_z: Tensor, base_tau: Tensor) -> Tensor:
        tx = base_tau[0].view(1, 1, 1)
        ty = base_tau[1].view(1, 1, 1)
        tz = base_tau[2].view(1, 1, 1)
        fx = self._conf_to_factor_1d(feat_x, tx)
        fy = self._conf_to_factor_1d(feat_y, ty)
        fz = self._conf_to_factor_1d(feat_z, tz)
        factor = torch.stack([fx, fy, fz], dim=-1)  # [B,K,3]
        if not self.axiswise:
            factor = factor.mean(dim=-1, keepdim=True).expand_as(factor)
        return factor

    def forward(
        self,
        sigma_logits: Optional[Tensor],
        feat_x: Tensor,
        feat_y: Tensor,
        feat_z: Tensor,
        base_tau: Tensor,
    ) -> Optional[Tensor]:
        """Return tau_factor [B,K,3] or None."""
        if not self.enable:
            return None

        factor_sigma = None
        factor_conf = None

        if self.source in {"sigma_head", "hybrid"}:
            if sigma_logits is None:
                raise RuntimeError("TemperatureAdapter requires sigma_logits but got None (source includes sigma_head).")
            factor_sigma = self._sigma_to_factor(sigma_logits)

        if self.source in {"simcc_conf", "hybrid"}:
            factor_conf = self._simcc_to_factor(feat_x, feat_y, feat_z, base_tau)

        if self.source == "sigma_head":
            return factor_sigma
        if self.source == "simcc_conf":
            return factor_conf

        alpha = float(self.hybrid_alpha)
        eps = self.eps
        assert factor_sigma is not None and factor_conf is not None
        return (factor_sigma.clamp(min=eps) ** alpha) * (factor_conf.clamp(min=eps) ** (1.0 - alpha))


class _FFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TokenSelfAttentionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        layer_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = _FFN(dim, mlp_ratio=mlp_ratio, drop=drop)
        self.ls1 = LayerScale(dim, layer_scale_init) if layer_scale_init > 0 else nn.Identity()
        self.ls2 = LayerScale(dim, layer_scale_init) if layer_scale_init > 0 else nn.Identity()
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        xn = self.norm1(x)
        y = self.attn(xn, xn, xn, need_weights=False)[0]
        x = x + self.drop(self.ls1(y))
        y = self.ffn(self.norm2(x))
        x = x + self.drop(self.ls2(y))
        return x


class MultiScaleDeformableCrossAttentionBlock(nn.Module):
    """Multi-scale deformable sampling-based cross-attn for joint tokens.

    This uses grid_sample (deployment-friendly) instead of a custom CUDA op.
    """

    def __init__(
        self,
        dim: int,
        feat_dim: int,
        num_levels: int = 2,
        num_points: int = 4,
        offset_range: Tuple[float, float] = (0.05, 0.05),
        coords_dim: int = 2,
        add_coords: bool = True,
        use_weights: bool = True,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        padding_mode: str = "border",
        align_corners: bool = False,
        detach_coords: bool = True,
        layer_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_levels = int(num_levels)
        self.num_points = int(num_points)
        self.coords_dim = int(coords_dim)
        assert self.coords_dim in (2, 3)

        self.add_coords = bool(add_coords)
        self.use_weights = bool(use_weights)
        self.padding_mode = str(padding_mode)
        self.align_corners = bool(align_corners)
        self.detach_coords = bool(detach_coords)

        self.register_buffer(
            "offset_range",
            torch.tensor(offset_range, dtype=torch.float32).view(1, 1, 1, 1, 2),
            persistent=False,
        )

        in_dim = dim + (self.coords_dim if self.add_coords else 0)
        self.norm_q = nn.LayerNorm(dim)

        self.offset_mlp = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, self.num_levels * self.num_points * 2),
        )
        nn.init.zeros_(self.offset_mlp[-1].weight)
        nn.init.zeros_(self.offset_mlp[-1].bias)

        if self.use_weights:
            self.weight_mlp = nn.Sequential(
                nn.Linear(in_dim, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, self.num_levels * self.num_points),
            )
            nn.init.zeros_(self.weight_mlp[-1].weight)
            nn.init.zeros_(self.weight_mlp[-1].bias)
        else:
            self.weight_mlp = None

        self.value_proj = nn.Identity() if feat_dim == dim else nn.Linear(feat_dim, dim, bias=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = _FFN(dim, mlp_ratio=mlp_ratio, drop=drop)
        self.ls1 = LayerScale(dim, layer_scale_init) if layer_scale_init > 0 else nn.Identity()
        self.ls2 = LayerScale(dim, layer_scale_init) if layer_scale_init > 0 else nn.Identity()
        self.drop = nn.Dropout(drop)

    def forward(self, q_joint: Tensor, feat_maps: Sequence[Tensor], coords_xyz: Tensor) -> Tensor:
        assert len(feat_maps) == self.num_levels, f"expected {self.num_levels} feat levels, got {len(feat_maps)}"

        qn = self.norm_q(q_joint)
        coords_in = coords_xyz.detach() if self.detach_coords else coords_xyz
        coords_mlp = coords_in[..., : self.coords_dim]
        mlp_in = torch.cat([qn, coords_mlp], dim=-1) if self.add_coords else qn

        B, K, _ = q_joint.shape
        offsets = self.offset_mlp(mlp_in).view(B, K, self.num_levels, self.num_points, 2)
        offsets = torch.tanh(offsets) * self.offset_range.to(device=offsets.device, dtype=offsets.dtype)

        ref_xy = coords_in[..., :2]
        sample_xy = (ref_xy.unsqueeze(2).unsqueeze(3) + offsets).clamp(0.0, 1.0)  # [B,K,L,M,2]

        # fast path if all levels share H,W 
        can_fast = True
        H0, W0 = feat_maps[0].shape[-2:]
        for fm in feat_maps[1:]:
            if fm.shape[-2:] != (H0, W0):
                can_fast = False
                break

        if can_fast:
            feats_all = _grid_sample_2d_multilevel(
                feat_maps,
                sample_xy,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )  # [B,K,L,M,C]

            if self.use_weights and self.weight_mlp is not None:
                w = F.softmax(self.weight_mlp(mlp_in), dim=-1).view(B, K, self.num_levels, self.num_points, 1)
                agg = (feats_all * w).sum(dim=3).sum(dim=2)  # [B,K,C]
            else:
                agg = feats_all.mean(dim=3).mean(dim=2)  # [B,K,C]
        else:
            if self.use_weights and self.weight_mlp is not None:
                w = F.softmax(self.weight_mlp(mlp_in), dim=-1).view(B, K, self.num_levels, self.num_points, 1)
                agg = 0.0
                for l in range(self.num_levels):
                    feats_l = _grid_sample_2d(
                        feat_maps[l],
                        sample_xy[:, :, l, :, :],
                        padding_mode=self.padding_mode,
                        align_corners=self.align_corners,
                    )  # [B,K,M,C]
                    agg = agg + (feats_l * w[:, :, l]).sum(dim=2)
            else:
                agg = 0.0
                inv = 1.0 / float(self.num_levels * self.num_points)
                for l in range(self.num_levels):
                    feats_l = _grid_sample_2d(
                        feat_maps[l],
                        sample_xy[:, :, l, :, :],
                        padding_mode=self.padding_mode,
                        align_corners=self.align_corners,
                    )
                    agg = agg + feats_l.sum(dim=2)
                agg = agg * inv

        v = self.value_proj(agg)
        q_joint = q_joint + self.drop(self.ls1(v))
        y = self.ffn(self.norm2(q_joint))
        q_joint = q_joint + self.drop(self.ls2(y))
        return q_joint


class JointRegDecoderLayerWithSigmaChannel(nn.Module):
    """One layer:
    1) self-attn over [registers + joints]
    2) multi-scale deformable cross-attn on joints only
    3) delta head on joints -> update coords outside
    4) sigma head on joints -> update sigma_logits (residual)
    """

    def __init__(
        self,
        dim: int,
        feat_dim: int,
        num_levels: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        num_points: int = 4,
        offset_range: Tuple[float, float] = (0.05, 0.05),
        coords_dim: int = 2,
        add_coords_to_cross: bool = True,
        add_coords_to_delta: bool = True,
        use_weights: bool = True,
        padding_mode: str = "border",
        align_corners: bool = False,
        detach_coords: bool = True,
        delta_mlp_layers: int = 2,
        sigma_mlp_layers: int = 2,
        layer_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.coords_dim = int(coords_dim)
        assert self.coords_dim in (2, 3)

        self.add_coords_to_delta = bool(add_coords_to_delta)
        self.detach_coords = bool(detach_coords)

        self.self_block = TokenSelfAttentionBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop,
            attn_drop=attn_drop,
            layer_scale_init=layer_scale_init,
        )

        self.cross_block = MultiScaleDeformableCrossAttentionBlock(
            dim=dim,
            feat_dim=feat_dim,
            num_levels=num_levels,
            num_points=num_points,
            offset_range=offset_range,
            coords_dim=self.coords_dim,
            add_coords=add_coords_to_cross,
            use_weights=use_weights,
            mlp_ratio=mlp_ratio,
            drop=drop,
            padding_mode=padding_mode,
            align_corners=align_corners,
            detach_coords=detach_coords,
            layer_scale_init=layer_scale_init,
        )

        delta_in_dim = dim + (self.coords_dim if self.add_coords_to_delta else 0)
        d_layers: List[nn.Module] = []
        in_dim = delta_in_dim
        for _ in range(max(0, int(delta_mlp_layers) - 1)):
            d_layers.append(nn.Linear(in_dim, dim))
            d_layers.append(nn.ReLU(inplace=True))
            in_dim = dim
        d_layers.append(nn.Linear(in_dim, 3))
        self.delta_head = nn.Sequential(*d_layers)
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

        s_layers: List[nn.Module] = []
        in_dim = dim
        for _ in range(max(0, int(sigma_mlp_layers) - 1)):
            s_layers.append(nn.Linear(in_dim, dim))
            s_layers.append(nn.ReLU(inplace=True))
            in_dim = dim
        s_layers.append(nn.Linear(in_dim, 3))
        self.sigma_head = nn.Sequential(*s_layers)
        nn.init.zeros_(self.sigma_head[-1].weight)
        nn.init.zeros_(self.sigma_head[-1].bias)

    def forward(
        self,
        tokens: Tensor,  # [B, R + K, D]
        feat_maps: Sequence[Tensor],  # list of [B, C, H, W]
        coords_xyz: Tensor,  # [B, K, 3]
        sigma_logits: Tensor,  # [B, K, 3]
        num_registers: int,
        num_joints: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        tokens = self.self_block(tokens)

        start_joint = num_registers
        end_joint = num_registers + num_joints
        q_joint = tokens[:, start_joint:end_joint, :]

        q_joint = self.cross_block(q_joint, feat_maps, coords_xyz)

        if start_joint > 0:
            tokens = torch.cat([tokens[:, :start_joint, :], q_joint], dim=1)
        else:
            tokens = q_joint

        coords_in = coords_xyz.detach() if self.detach_coords else coords_xyz
        if self.add_coords_to_delta:
            delta_in = torch.cat([q_joint, coords_in[..., : self.coords_dim]], dim=-1)
        else:
            delta_in = q_joint
        delta = self.delta_head(delta_in)

        sigma_logits = sigma_logits + self.sigma_head(q_joint)
        return tokens, delta, sigma_logits


@MODELS.register_module()
class RTMCCIPRHead3DJointFormer(RTMCCHead3D):
    DEFAULT_BONE_PAIRS = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ]

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        input_size: Tuple[int, int, int],
        in_featuremap_size: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
        final_layer_kernel_size: int = 1,
        gau_cfg: ConfigType = dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn="ReLU",
            use_rel_bias=False,
            pos_enc=False,
        ),
        loss: ConfigType = dict(type="KLDiscretLoss", use_target_weight=True),
        decoder: OptConfigType = None,
        init_cfg: OptConfigType = None,
        with_gau: bool = False,
        mlp_with_conv: bool = False,
        map_type: str = "softmax",
        temperature: Union[float, Tuple[float, float, float]] = 1.0,
        learnable_temperature: bool = False,
        output_sigma: bool = False,
        sigma_temperature_cfg: Optional[ConfigType] = None,
        deploy: bool = False,
        deploy_output: str = "kpt",
        refine_cfg: ConfigType = dict(),
        linspace_denominator: str = "L",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            input_size=input_size,
            in_featuremap_size=in_featuremap_size,
            simcc_split_ratio=simcc_split_ratio,
            final_layer_kernel_size=final_layer_kernel_size,
            gau_cfg=gau_cfg,
            loss=loss,
            decoder=decoder,
            init_cfg=init_cfg,
            with_gau=with_gau,
            mlp_with_conv=mlp_with_conv,
        )

        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)
        D = int(self.input_size[2] * self.simcc_split_ratio)
        self.ipr_module = SimCCToKeypoint3DPlus(
            feat_w=W,
            feat_h=H,
            feat_d=D,
            map_type=map_type,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
            linspace_denominator=linspace_denominator,
        )

        self.output_sigma = bool(output_sigma)
        self.deploy = bool(deploy)
        self.deploy_output = str(deploy_output)

        sigma_temperature_cfg = dict(sigma_temperature_cfg or {})
        self.temp_adapter = TemperatureAdapter(
            enable=bool(_cfg_get(sigma_temperature_cfg, "enable", False)),
            source=str(_cfg_get(sigma_temperature_cfg, "source", "sigma_head")),
            conf_type=str(_cfg_get(sigma_temperature_cfg, "conf_type", "entropy")),
            factor_range=tuple(_cfg_get(sigma_temperature_cfg, "factor_range", (0.70, 1.60))),
            axiswise=bool(_cfg_get(sigma_temperature_cfg, "axiswise", True)),
            detach=bool(_cfg_get(sigma_temperature_cfg, "detach", False)),
            sigma_mode=str(_cfg_get(sigma_temperature_cfg, "sigma_mode", "sigmoid")),
            hybrid_alpha=float(_cfg_get(sigma_temperature_cfg, "hybrid_alpha", 0.5)),
            eps=float(_cfg_get(sigma_temperature_cfg, "eps", 1e-8)),
        )

        rcfg = dict(refine_cfg or {})
        self.refine_cfg = rcfg

        # sigma refinement & NLL supervision
        self.enable_sigma_refine = bool(_cfg_get(rcfg, "enable_sigma_refine", True))
        sigma_nll_cfg = dict(_cfg_get(rcfg, "sigma_nll_cfg", {}) or {})
        self.sigma_nll_enable = bool(_cfg_get(sigma_nll_cfg, "enable", True))
        self.sigma_nll_type = str(_cfg_get(sigma_nll_cfg, "type", "laplace")).lower()
        assert self.sigma_nll_type in ("laplace", "gaussian")
        self.sigma_nll_weight = float(_cfg_get(sigma_nll_cfg, "weight", 1.0))
        self.sigma_min = float(_cfg_get(sigma_nll_cfg, "min_sigma", 1e-4))
        self.sigma_nll_eps = float(_cfg_get(sigma_nll_cfg, "eps", 1e-8))

        # multi-scale features
        refine_feat_indices = _cfg_get(rcfg, "refine_feat_indices", [-2, -1])
        if isinstance(refine_feat_indices, int):
            refine_feat_indices = [int(refine_feat_indices)]
        self.refine_feat_indices: List[int] = [int(i) for i in refine_feat_indices]

        self.num_refine_levels = int(_cfg_get(rcfg, "num_refine_levels", len(self.refine_feat_indices)))
        self.refine_feat_indices = self.refine_feat_indices[: self.num_refine_levels]
        self.num_refine_levels = len(self.refine_feat_indices)

        self.proj_channels = int(_cfg_get(rcfg, "proj_channels", 128))
        self.padding_mode = str(_cfg_get(rcfg, "padding_mode", "border"))
        self.align_corners_refine = bool(_cfg_get(rcfg, "align_corners", False))
        self.detach_coords_ref = bool(_cfg_get(rcfg, "detach_coarse_for_sample", True))

        self.deform_num_points = int(_cfg_get(rcfg, "deform_num_points", 4))
        self.use_deform_weights = bool(_cfg_get(rcfg, "use_deform_weights", True))
        offset_range = _cfg_get(rcfg, "offset_range", (0.05, 0.05))
        if isinstance(offset_range, (int, float)):
            offset_range = (float(offset_range), float(offset_range))
        self.offset_range = (float(offset_range[0]), float(offset_range[1]))

        self.coords_dim = int(_cfg_get(rcfg, "coords_dim", 2))
        assert self.coords_dim in (2, 3)
        self.refine_z = bool(_cfg_get(rcfg, "refine_z", True))
        self.use_inv_sigmoid = bool(_cfg_get(rcfg, "use_inv_sigmoid", True))

        delta_scale = _cfg_get(rcfg, "delta_scale", (0.03, 0.03, 0.05))
        if isinstance(delta_scale, (int, float)):
            delta_scale = (float(delta_scale),) * 3
        self.register_buffer(
            "delta_scale",
            torch.tensor(delta_scale, dtype=torch.float32).view(1, 1, 3),
            persistent=False,
        )

        # decoder params
        self.decoder_layers = int(_cfg_get(rcfg, "decoder_layers", _cfg_get(rcfg, "num_refine_iters", 2)))
        self.embed_dim = int(_cfg_get(rcfg, "decoder_embed_dim", 256))
        self.decoder_heads = int(_cfg_get(rcfg, "decoder_heads", 4))
        self.decoder_mlp_ratio = float(_cfg_get(rcfg, "decoder_mlp_ratio", 4.0))
        self.decoder_drop = float(_cfg_get(rcfg, "decoder_drop", 0.0))
        self.decoder_attn_drop = float(_cfg_get(rcfg, "decoder_attn_drop", 0.0))
        self.decoder_delta_mlp_layers = int(_cfg_get(rcfg, "decoder_delta_mlp_layers", 2))
        self.decoder_sigma_mlp_layers = int(_cfg_get(rcfg, "decoder_sigma_mlp_layers", 2))
        self.layer_scale_init = float(_cfg_get(rcfg, "layer_scale_init_value", 1e-5))

        # register tokens
        self.num_register_tokens = int(_cfg_get(rcfg, "num_register_tokens", 2))
        self.use_register_tokens = bool(_cfg_get(rcfg, "use_register_tokens", True)) and self.num_register_tokens > 0
        if not self.use_register_tokens:
            self.num_register_tokens = 0

        self.joint_embed = nn.Parameter(torch.zeros(1, self.out_channels, self.embed_dim))
        nn.init.trunc_normal_(self.joint_embed, std=0.02)

        if self.use_register_tokens:
            self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, self.embed_dim))
            nn.init.trunc_normal_(self.register_tokens, std=0.02)
        else:
            self.register_tokens = None

        self.use_coord_embed = bool(_cfg_get(rcfg, "use_coord_embed", True))
        self.coord_embed = nn.Linear(self.coords_dim, self.embed_dim, bias=True) if self.use_coord_embed else None

        # sigma conditioning
        self.sigma_cond_dim = int(_cfg_get(rcfg, "sigma_cond_dim", 16))
        if self.sigma_cond_dim > 0:
            self.sigma_cond_proj = nn.Linear(3, self.sigma_cond_dim, bias=True)
            self.sigma_cond_norm = nn.LayerNorm(self.sigma_cond_dim)
            self.joint_fuse = nn.Linear(self.embed_dim + self.sigma_cond_dim, self.embed_dim, bias=True)
        else:
            self.sigma_cond_proj = None
            self.sigma_cond_norm = None
            self.joint_fuse = None

        need_sigma_logits = (
            (self.temp_adapter.enable and self.temp_adapter.source in {"sigma_head", "hybrid"})
            or self.output_sigma
            or self.enable_sigma_refine
            or (self.sigma_nll_enable and self.sigma_nll_weight > 0)
        )
        if need_sigma_logits:
            base_in_sig = self.in_channels[-1] if isinstance(self.in_channels, (list, tuple)) else self.in_channels
            base_in_sig = int(base_in_sig)
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.sigma_conv = nn.Conv2d(base_in_sig, self.out_channels * 3, kernel_size=1, bias=True)
            nn.init.zeros_(self.sigma_conv.weight)
            nn.init.zeros_(self.sigma_conv.bias)
        else:
            self.gap = None
            self.sigma_conv = None

        # feature map builders
        self.post_upsample_conv = bool(_cfg_get(rcfg, "post_upsample_conv", True))
        refine_upsample_scales = _cfg_get(rcfg, "refine_upsample_scales", None)
        if refine_upsample_scales is None:
            self.refine_upsample_scales = [1.0 for _ in range(self.num_refine_levels)]
        else:
            if isinstance(refine_upsample_scales, (int, float)):
                self.refine_upsample_scales = [float(refine_upsample_scales) for _ in range(self.num_refine_levels)]
            else:
                self.refine_upsample_scales = [float(s) for s in list(refine_upsample_scales)]
                if len(self.refine_upsample_scales) < self.num_refine_levels:
                    self.refine_upsample_scales += [1.0] * (self.num_refine_levels - len(self.refine_upsample_scales))
                self.refine_upsample_scales = self.refine_upsample_scales[: self.num_refine_levels]

        self.upsample_mode = str(_cfg_get(rcfg, "upsample_mode", "bilinear"))

        cin_list = rcfg.get("refine_in_channels_list", None)
        if cin_list is None:
            if isinstance(self.in_channels, (list, tuple)):
                cin_list = [int(c) for c in self.in_channels]
            else:
                cin_list = [int(self.in_channels)] * self.num_refine_levels
        else:
            cin_list = [int(c) for c in list(cin_list)]
        if len(cin_list) < self.num_refine_levels:
            cin_list += [cin_list[-1]] * (self.num_refine_levels - len(cin_list))
        cin_list = cin_list[: self.num_refine_levels]

        self.refine_proj_list = nn.ModuleList()
        self.refine_post_list = nn.ModuleList()
        for i in range(self.num_refine_levels):
            cin = cin_list[i]
            self.refine_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(cin, self.proj_channels, kernel_size=1, bias=True),
                    nn.ReLU(inplace=True),
                )
            )
            if self.post_upsample_conv:
                self.refine_post_list.append(
                    nn.Sequential(
                        nn.Conv2d(self.proj_channels, self.proj_channels, kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                self.refine_post_list.append(nn.Identity())

        # decoder stack
        self.refine_layers = nn.ModuleList(
            [
                JointRegDecoderLayerWithSigmaChannel(
                    dim=self.embed_dim,
                    feat_dim=self.proj_channels,
                    num_levels=self.num_refine_levels,
                    num_heads=self.decoder_heads,
                    mlp_ratio=self.decoder_mlp_ratio,
                    drop=self.decoder_drop,
                    attn_drop=self.decoder_attn_drop,
                    num_points=self.deform_num_points,
                    offset_range=self.offset_range,
                    coords_dim=self.coords_dim,
                    add_coords_to_cross=bool(_cfg_get(rcfg, "add_coords_to_cross_attn", True)),
                    add_coords_to_delta=bool(_cfg_get(rcfg, "add_coords_to_delta", True)),
                    use_weights=self.use_deform_weights,
                    padding_mode=self.padding_mode,
                    align_corners=self.align_corners_refine,
                    detach_coords=self.detach_coords_ref,
                    delta_mlp_layers=self.decoder_delta_mlp_layers,
                    sigma_mlp_layers=self.decoder_sigma_mlp_layers,
                    layer_scale_init=self.layer_scale_init,
                )
                for _ in range(max(1, self.decoder_layers))
            ]
        )

        # loss weights
        self.refine_loss_weight = float(_cfg_get(rcfg, "refine_loss_weight", 1.0))
        self.coarse_loss_weight = float(_cfg_get(rcfg, "coarse_loss_weight", 0.5))

        # deep supervision
        ds_cfg = dict(_cfg_get(rcfg, "deep_supervision", {}) or {})
        self.ds_enable = bool(_cfg_get(ds_cfg, "enable", False))
        self.ds_weight = float(_cfg_get(ds_cfg, "weight", 0.0))
        self.ds_strategy = str(_cfg_get(ds_cfg, "strategy", "linear")).lower()
        self.ds_detach = bool(_cfg_get(ds_cfg, "detach", False))
        self.ds_include_final = bool(_cfg_get(ds_cfg, "include_final", False))
        assert self.ds_strategy in {"linear", "exp", "uniform"}

        # bone loss
        self.bone_loss_weight = float(_cfg_get(rcfg, "bone_loss_weight", 0.1))
        self.bone_loss_3d_only = bool(_cfg_get(rcfg, "bone_loss_3d_only", False))
        bone_pairs = _cfg_get(rcfg, "bone_pairs", None) or self.DEFAULT_BONE_PAIRS
        self.bone_pairs: List[Tuple[int, int]] = [(int(a), int(b)) for a, b in bone_pairs]
        self.bone_min_gt_len = float(_cfg_get(rcfg, "bone_min_gt_len", 0.01))
        self.bone_huber_delta = float(_cfg_get(rcfg, "bone_huber_delta", 0.05))
        self.bone_use_relative = bool(_cfg_get(rcfg, "bone_use_relative", True))
        self.bone_clamp_per_bone = float(_cfg_get(rcfg, "bone_clamp_per_bone", 10.0))

        # OKS loss
        self.use_oks_loss = bool(_cfg_get(rcfg, "use_oks_loss", False))
        self.oks_loss_weight = float(_cfg_get(rcfg, "oks_loss_weight", 0.0))
        self.oks_loss_type = str(_cfg_get(rcfg, "oks_loss_type", "1moks")).lower()
        assert self.oks_loss_type in ("1moks", "neglog")
        self.coord_is_normalized = bool(_cfg_get(rcfg, "coord_is_normalized", True))
        self.oks_from_bbox = bool(_cfg_get(rcfg, "oks_from_bbox", False))
        self.oks_fallback_kpt_bbox = bool(_cfg_get(rcfg, "oks_fallback_kpt_bbox", True))
        self.oks_eps = float(_cfg_get(rcfg, "oks_eps", 1e-6))

        # OKS loss 
        self.oks_use_pred_sigma = bool(_cfg_get(rcfg, "oks_use_pred_sigma", False))
        self.oks_sigma_fuse = str(_cfg_get(rcfg, "oks_sigma_fuse", "add")).lower()
        assert self.oks_sigma_fuse in ("add", "mul")
        self.oks_focal_gamma = float(_cfg_get(rcfg, "oks_focal_gamma", 0.0))

        # optional: apply oks loss to coarse / deep supervision
        self.oks_on_coarse = bool(_cfg_get(rcfg, "oks_on_coarse", False))
        self.oks_on_ds = bool(_cfg_get(rcfg, "oks_on_ds", False))
        self.oks_aux_weight = float(_cfg_get(rcfg, "oks_aux_weight", 0.5))


        sigmas = [
            0.87,
            0.62,
            0.35,
            0.25,
            0.25,
            0.39,
            0.25,
            0.25,
            0.25,
            0.39,
            0.25,
            0.25,
            0.25,
            0.25,
            0.25,
            0.25,
            0.25,
            0.39,
            0.25,
            0.25,
            0.25,
        ]
        self.register_buffer("oks_sigmas", torch.tensor(sigmas, dtype=torch.float32) / 10.0, persistent=False)

    @staticmethod
    def _inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
        x = x.clamp(min=eps, max=1.0 - eps)
        return torch.log(x / (1.0 - x))

    def _select_refine_feats_multi(self, feats: Tuple[Tensor, ...]) -> List[Tensor]:
        return [feats[idx] for idx in self.refine_feat_indices]

    def _build_feat_maps_multi(self, refine_feats_list: Sequence[Tensor]) -> List[Tensor]:
        assert len(refine_feats_list) == self.num_refine_levels
        feat_maps: List[Tensor] = []
        for i, x in enumerate(refine_feats_list):
            fm = self.refine_proj_list[i](x)
            s = float(self.refine_upsample_scales[i]) if i < len(self.refine_upsample_scales) else 1.0
            if s != 1.0:
                fm = F.interpolate(
                    fm,
                    scale_factor=s,
                    mode=self.upsample_mode,
                    align_corners=self.align_corners_refine if self.upsample_mode in ("bilinear", "bicubic") else None,
                )
            fm = self.refine_post_list[i](fm)
            feat_maps.append(fm)
        return feat_maps

    def _predict_sigma_logits(self, raw_feats: Tensor) -> Optional[Tensor]:
        if self.sigma_conv is None or self.gap is None:
            return None
        x = self.gap(raw_feats)
        return self.sigma_conv(x).reshape(x.size(0), self.out_channels, 3)

    def _sigma_from_logits(self, sigma_logits: Tensor) -> Tensor:
        return F.softplus(sigma_logits) + float(self.sigma_min)

    def _nll_per_axis(self, err: Tensor, sigma: Tensor) -> Tensor:
        eps = float(self.sigma_nll_eps)
        sigma = sigma.clamp(min=eps)

        if self.sigma_nll_type == "laplace":
            nll = err.abs() / sigma + torch.log(2.0 * sigma)
            nll = nll - math.log(2.0 * eps)
            return nll.clamp_min(0.0)

        var = (sigma * sigma).clamp(min=eps * eps)
        nll = 0.5 * (err * err / var + torch.log(2.0 * math.pi * var))
        nll = nll - 0.5 * math.log(2.0 * math.pi * (eps * eps))
        return nll.clamp_min(0.0)

    def _fuse_sigma_into_tokens(self, tokens: Tensor, sig: Tensor, R: int, K: int) -> Tensor:
        if self.sigma_cond_dim <= 0 or self.sigma_cond_proj is None or self.joint_fuse is None:
            return tokens
        sigma = self._sigma_from_logits(sig)
        s = self.sigma_cond_proj(sigma)
        if self.sigma_cond_norm is not None:
            s = self.sigma_cond_norm(s)
        if R > 0:
            reg = tokens[:, :R, :]
            jt = tokens[:, R : R + K, :]
            jt = self.joint_fuse(torch.cat([jt, s], dim=-1))
            return torch.cat([reg, jt], dim=1)
        return self.joint_fuse(torch.cat([tokens, s], dim=-1))

    def _forward_coarse(self, feats: Tuple[Tensor, ...]):
        feat_x, feat_y, feat_z = super().forward(feats)
        heatmaps = torch.cat([feat_x, feat_y, feat_z], dim=1)

        raw_feats = feats[-1]
        refine_feats_list = self._select_refine_feats_multi(feats)
        sigma_logits = self._predict_sigma_logits(raw_feats)

        base_tau = self.ipr_module.get_base_tau(device=feat_x.device, dtype=feat_x.dtype)
        tau_factor = self.temp_adapter(sigma_logits, feat_x, feat_y, feat_z, base_tau) if self.temp_adapter.enable else None

        pred_x, pred_y, pred_z = self.ipr_module(feat_x, feat_y, feat_z, tau_factor=tau_factor)
        coarse = torch.cat([pred_x, pred_y, pred_z], dim=-1)  # [B,K,3]
        return coarse, heatmaps, raw_feats, refine_feats_list, (feat_x, feat_y, feat_z), sigma_logits

    def _init_tokens(self, coords_xyz: Tensor, sigma_logits: Tensor) -> Tensor:
        """tokens = [registers(optional), joints]; sigma is injected as a channel."""
        B, K, _ = coords_xyz.shape
        coords_in = coords_xyz.detach() if self.detach_coords_ref else coords_xyz

        joint_tokens = self.joint_embed.expand(B, -1, -1)
        if self.use_coord_embed and self.coord_embed is not None:
            joint_tokens = joint_tokens + self.coord_embed(coords_in[..., : self.coords_dim])

        if self.sigma_cond_dim > 0 and self.sigma_cond_proj is not None and self.joint_fuse is not None:
            sigma = self._sigma_from_logits(sigma_logits)
            s = self.sigma_cond_proj(sigma)
            if self.sigma_cond_norm is not None:
                s = self.sigma_cond_norm(s)
            joint_tokens = self.joint_fuse(torch.cat([joint_tokens, s], dim=-1))

        if self.use_register_tokens and self.register_tokens is not None:
            reg = self.register_tokens.expand(B, -1, -1)
            return torch.cat([reg, joint_tokens], dim=1)
        return joint_tokens

    def _forward_refined(
        self,
        coarse: Tensor,
        refine_feats_list: Sequence[Tensor],
        sigma_logits: Tensor,
        return_intermediates: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[List[Tensor]]]:
        feat_maps = self._build_feat_maps_multi(refine_feats_list)

        coords = coarse
        sig = sigma_logits
        tokens = self._init_tokens(coords, sig)

        R = self.num_register_tokens if self.use_register_tokens else 0
        K = self.out_channels
        delta_scale = self.delta_scale.to(device=coarse.device, dtype=coarse.dtype)

        coords_list: Optional[List[Tensor]] = [] if return_intermediates else None

        for layer in self.refine_layers:
            tokens, delta, sig = layer(tokens, feat_maps, coords, sig, num_registers=R, num_joints=K)
            delta = torch.tanh(delta) * delta_scale

            if not self.refine_z:
                delta = delta.clone()
                delta[..., 2] = 0.0

            if self.use_inv_sigmoid:
                coords = torch.sigmoid(self._inverse_sigmoid(coords) + delta)
            else:
                coords = (coords + delta).clamp(0.0, 1.0)

            tokens = self._fuse_sigma_into_tokens(tokens, sig, R, K)

            if return_intermediates:
                assert coords_list is not None
                coords_list.append(coords)

        return coords, sig, coords_list

    def _forward(self, feats: Tuple[Tensor, ...]):
        coarse, _, raw_feats, refine_feats_list, (feat_x, feat_y, feat_z), sigma_logits = self._forward_coarse(feats)

        if sigma_logits is None:
            refined = self._forward_refined(coarse, refine_feats_list, torch.zeros_like(coarse))[0]
        else:
            refined = self._forward_refined(coarse, refine_feats_list, sigma_logits)[0]

        if self.deploy_output == "kpt":
            return refined[..., :3]
        if self.deploy_output == "feat":
            return feat_x, feat_y, feat_z, raw_feats, refine_feats_list
        return refined[..., :3]

    def forward(self, feats: Tuple[Tensor, ...]):
        coarse, heatmaps, raw_feats, refine_feats_list, (feat_x, feat_y, feat_z), sigma_logits = self._forward_coarse(
            feats
        )

        if sigma_logits is None:
            refined, _, _ = self._forward_refined(coarse, refine_feats_list, torch.zeros_like(coarse))
            sigma_logits_ref = None
        else:
            refined, sigma_logits_ref, _ = self._forward_refined(coarse, refine_feats_list, sigma_logits)

        output = refined
        if self.output_sigma and (sigma_logits_ref is not None):
            sigma = self._sigma_from_logits(sigma_logits_ref)
            output = torch.cat([output, sigma], dim=-1)

        if self.deploy:
            if self.deploy_output == "kpt":
                return output[..., :3]
            if self.deploy_output == "feat":
                return feat_x, feat_y, feat_z, raw_feats, refine_feats_list
        return output, heatmaps

    def predict(
        self,
        feats: Union[Tuple[Tensor, ...], List[Tuple[Tensor, ...]]],
        batch_data_samples: OptSampleList,
        test_cfg: ConfigType = {},
    ):
        if self.deploy:
            if self.deploy_output == "feat":
                feat_x, feat_y, feat_z, raw_feats, refine_feats_list = feats  
                sigma_logits = self._predict_sigma_logits(raw_feats)
                base_tau = self.ipr_module.get_base_tau(device=feat_x.device, dtype=feat_x.dtype)
                tau_factor = self.temp_adapter(sigma_logits, feat_x, feat_y, feat_z, base_tau) if self.temp_adapter.enable else None
                pred_x, pred_y, pred_z = self.ipr_module(feat_x, feat_y, feat_z, tau_factor=tau_factor)
                coarse = torch.cat([pred_x, pred_y, pred_z], dim=-1)

                if sigma_logits is None:
                    batch_coords = self._forward_refined(coarse, refine_feats_list, torch.zeros_like(coarse))[0]
                else:
                    batch_coords = self._forward_refined(coarse, refine_feats_list, sigma_logits)[0]
            else:
                batch_coords = feats  
        else:
            if test_cfg.get("flip_test", False):
                assert isinstance(feats, list) and len(feats) == 2
                flip_indices = batch_data_samples[0].metainfo["flip_indices"]
                input_size = batch_data_samples[0].metainfo["input_size"]
                _feats, _feats_flip = feats

                out, _ = self.forward(_feats)
                out_f, _ = self.forward(_feats_flip)

                coords = out[..., :3]
                coords_f = flip_coordinates(
                    out_f[..., :3],
                    flip_indices=flip_indices,
                    shift_coords=test_cfg.get("shift_coords", True),
                    input_size=input_size,
                )
                batch_coords = (coords + coords_f) * 0.5
            else:
                out, _ = self.forward(feats)  
                batch_coords = out[..., :3]

        batch_coords = batch_coords.unsqueeze(1)  # [B,1,K,3]
        preds = self.decode(batch_coords)
        return preds

    @staticmethod
    def _bone_loss(
        pred: Tensor,
        gt: Tensor,
        vis: Tensor,
        pairs: List[Tuple[int, int]],
        min_gt_len: float = 0.01,
        huber_delta: float = 0.05,
        use_relative: bool = True,
        clamp_per_bone: float = 10.0,
        eps: float = 1e-8,
    ) -> Tensor:
        if pred.numel() == 0:
            return pred.sum() * 0.0

        device, dtype = pred.device, pred.dtype
        idx_a = torch.tensor([a for a, _ in pairs], device=device, dtype=torch.long)
        idx_b = torch.tensor([b for _, b in pairs], device=device, dtype=torch.long)

        pred_a = pred.index_select(1, idx_a)
        pred_b = pred.index_select(1, idx_b)
        gt_a = gt.index_select(1, idx_a)
        gt_b = gt.index_select(1, idx_b)

        lp = torch.norm(pred_a - pred_b, dim=-1)
        lg = torch.norm(gt_a - gt_b, dim=-1)

        m = vis.index_select(1, idx_a) & vis.index_select(1, idx_b)
        m = m & (lg > float(min_gt_len))
        if m.sum() == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)

        if use_relative:
            denom = torch.clamp(lg, min=float(min_gt_len))
            e = (lp - lg).abs() / denom
        else:
            e = (lp - lg).abs()

        e = e[m].to(dtype=dtype)
        d = torch.tensor(float(huber_delta), device=device, dtype=dtype)
        huber = torch.where(e <= d, 0.5 * e * e, d * (e - 0.5 * d))
        if clamp_per_bone is not None and clamp_per_bone > 0:
            huber = torch.clamp(huber, max=float(clamp_per_bone))
        return huber.mean()

    @staticmethod
    def _kpt_to_bbox_area(gt_xy_px: Tensor, vis: Tensor, eps: float = 1e-6) -> Tensor:
        vis = vis.to(dtype=torch.bool)
        x = gt_xy_px[..., 0]
        y = gt_xy_px[..., 1]
        big = torch.full_like(x, 1e9)
        small = torch.full_like(x, -1e9)
        x_min = torch.where(vis, x, big).min(dim=1).values
        y_min = torch.where(vis, y, big).min(dim=1).values
        x_max = torch.where(vis, x, small).max(dim=1).values
        y_max = torch.where(vis, y, small).max(dim=1).values
        w = (x_max - x_min).clamp(min=eps)
        h = (y_max - y_min).clamp(min=eps)
        return (w * h).clamp(min=eps)

    def _get_oks_area(self, batch_data_samples: OptSampleList, gt_xy_px: Tensor, vis: Tensor) -> Tensor:
        areas: List[Tensor] = []
        for i, data in enumerate(batch_data_samples):
            area_i = None
            if self.oks_from_bbox and hasattr(data, "gt_instances") and data.gt_instances is not None:
                if hasattr(data.gt_instances, "bboxes") and data.gt_instances.bboxes is not None:
                    b = data.gt_instances.bboxes
                    if isinstance(b, torch.Tensor) and b.numel() >= 4:
                        bb = b.to(gt_xy_px.device)[0]
                        w = (bb[2] - bb[0]).clamp(min=self.oks_eps)
                        h = (bb[3] - bb[1]).clamp(min=self.oks_eps)
                        area_i = w * h
            if area_i is None:
                if self.oks_fallback_kpt_bbox:
                    area_i = self._kpt_to_bbox_area(gt_xy_px[i : i + 1], vis[i : i + 1], eps=self.oks_eps)[0]
                else:
                    area_i = torch.tensor(1.0, device=gt_xy_px.device, dtype=gt_xy_px.dtype)
            areas.append(area_i)
        return torch.stack(areas, dim=0)

    def _oks(self, pred_xy_px: Tensor, gt_xy_px: Tensor, vis: Tensor, area: Tensor) -> Tensor:
        B, K, _ = pred_xy_px.shape
        vis_f = vis.to(dtype=pred_xy_px.dtype)
        dx = pred_xy_px[..., 0] - gt_xy_px[..., 0]
        dy = pred_xy_px[..., 1] - gt_xy_px[..., 1]
        d2 = dx * dx + dy * dy
        vars_ = (2.0 * self.oks_sigmas.to(dtype=pred_xy_px.dtype)).pow(2).view(1, K).clamp(min=self.oks_eps)
        area = area.view(B, 1).clamp(min=self.oks_eps)
        e = d2 / vars_ / area / 2.0
        oks = (torch.exp(-e) * vis_f).sum(dim=1) / vis_f.sum(dim=1).clamp(min=1.0)
        return oks

    def _ds_weights(self, num_stages: int) -> List[float]:
        """Return per-stage weights for deep supervision."""
        if num_stages <= 0:
            return []
        w_total = float(self.ds_weight)
        if w_total <= 0:
            return [0.0 for _ in range(num_stages)]

        if self.ds_strategy == "uniform":
            return [w_total / float(num_stages) for _ in range(num_stages)]

        if self.ds_strategy == "exp":
            base = 2.0
            raw = [base ** float(i) for i in range(num_stages)]
        else:
            raw = [float(i + 1) for i in range(num_stages)]

        s = sum(raw)
        if s <= 0:
            return [0.0 for _ in range(num_stages)]
        return [w_total * (r / s) for r in raw]

    def loss(
        self,
        inputs: Tuple[Tensor, ...],
        batch_data_samples: OptSampleList,
        train_cfg: ConfigType = {},
    ) -> dict:
        coarse, _, raw_feats, refine_feats_list, _, sigma_logits0 = self._forward_coarse(inputs)

        need_intermediate = bool(self.ds_enable and self.ds_weight > 0 and len(self.refine_layers) > 1)

        if sigma_logits0 is None:
            refined_xyz, _, coords_list = self._forward_refined(
                coarse, refine_feats_list, torch.zeros_like(coarse), return_intermediates=need_intermediate
            )
            sigma_logits_ref = None
        else:
            refined_xyz, sigma_logits_ref, coords_list = self._forward_refined(
                coarse, refine_feats_list, sigma_logits0, return_intermediates=need_intermediate
            )

        pred_ref_xyz = refined_xyz[..., :3]
        pred_coarse_xyz = coarse[..., :3]
        device = pred_ref_xyz.device

        keypoint_weights = torch.cat(
            [d.gt_instance_labels.keypoint_weights for d in batch_data_samples],
            dim=0,
        ).to(device).unsqueeze(-1)  # [B,K,1]

        label_2d_list = []
        label_depth_list = []
        label_depth_id_list = []
        for i, data in enumerate(batch_data_samples):
            lbl = data.gt_instance_labels.keypoint_labels
            label_2d_list.append(lbl[..., :2])
            if lbl.shape[-1] >= 3:
                label_depth_list.append(lbl[..., 2:3])
                label_depth_id_list.append(i)

        label_2d = torch.cat(label_2d_list, dim=0).to(device)
        Bd = len(label_depth_id_list)
        keypoint3d_ratio = Bd / float(len(batch_data_samples))

        if Bd > 0:
            label_depth = torch.cat(label_depth_list, dim=0).to(device)
            label_depth_id = torch.tensor(label_depth_id_list, dtype=torch.long, device=device)
            valid_depth_weights = torch.index_select(keypoint_weights, 0, label_depth_id)
        else:
            label_depth = None
            label_depth_id = None
            valid_depth_weights = None

        def _depth_branch(pred_xyz: Tensor):
            if Bd > 0:
                assert label_depth is not None and label_depth_id is not None and valid_depth_weights is not None
                depth_pred = torch.index_select(pred_xyz, 0, label_depth_id)[..., 2:3]
                return depth_pred, label_depth, valid_depth_weights
            dummy_pred = pred_xyz[:1, :, 2:3]
            dummy_label = torch.zeros_like(dummy_pred)
            dummy_weight = torch.zeros_like(keypoint_weights[:1])
            return dummy_pred, dummy_label, dummy_weight

        losses: Dict[str, Tensor] = {}
        losses["kpt3d_ratio"] = torch.tensor([keypoint3d_ratio], device=device)

        # refined loss
        ref_depth_pred, ref_depth_label, ref_depth_w = _depth_branch(pred_ref_xyz)
        ref_inputs = [pred_ref_xyz[..., :2], ref_depth_pred]
        ref_targets = [label_2d, ref_depth_label]
        ref_weights = [keypoint_weights, ref_depth_w]
        ref_loss_list = self.loss_module(ref_inputs, ref_targets, ref_weights)
        losses["loss_kpt2d"] = ref_loss_list[0] * self.refine_loss_weight
        losses["loss_depth"] = ref_loss_list[1] * self.refine_loss_weight

        # deep supervision 
        if need_intermediate and coords_list is not None and len(coords_list) > 1:
            ds_coords = coords_list if self.ds_include_final else coords_list[:-1]
            ds_w = self._ds_weights(len(ds_coords))
            for i, (c_i, w_i) in enumerate(zip(ds_coords, ds_w)):
                if w_i <= 0:
                    continue
                c_i = c_i.detach() if self.ds_detach else c_i
                p_i = c_i[..., :3]
                d_pred, d_label, d_w = _depth_branch(p_i)
                i_inputs = [p_i[..., :2], d_pred]
                i_targets = [label_2d, d_label]
                i_weights = [keypoint_weights, d_w]
                l_list = self.loss_module(i_inputs, i_targets, i_weights)
                losses[f"loss_kpt2d_ds{i+1}"] = l_list[0] * float(w_i)
                losses[f"loss_depth_ds{i+1}"] = l_list[1] * float(w_i)

        # sigma NLL supervision
        if self.sigma_nll_enable and self.sigma_nll_weight > 0 and (sigma_logits_ref is not None):
            sigma = self._sigma_from_logits(sigma_logits_ref)  # [B,K,3]

            err_xy = pred_ref_xyz[..., :2] - label_2d
            sigma_xy = sigma[..., :2]
            m_xy = (keypoint_weights[..., 0] > 0).to(dtype=err_xy.dtype).unsqueeze(-1)

            nll_xy = self._nll_per_axis(err_xy, sigma_xy)
            nll_xy = (nll_xy * m_xy).sum() / (m_xy.sum().clamp(min=1.0) * 2.0)

            if Bd > 0:
                assert label_depth is not None and label_depth_id is not None and valid_depth_weights is not None
                pred_z = torch.index_select(pred_ref_xyz, 0, label_depth_id)[..., 2:3]
                sigma_z = torch.index_select(sigma, 0, label_depth_id)[..., 2:3]
                gt_z = label_depth
                m_z = (valid_depth_weights[..., 0] > 0).to(dtype=pred_z.dtype).unsqueeze(-1)
                err_z = pred_z - gt_z
                nll_z = self._nll_per_axis(err_z, sigma_z)
                nll_z = (nll_z * m_z).sum() / (m_z.sum().clamp(min=1.0) * 1.0)
                nll = 0.9 * nll_xy + 0.1 * nll_z
            else:
                nll = nll_xy

            losses["loss_sigma_nll"] = nll * self.sigma_nll_weight

        # bone loss
        if self.bone_loss_weight > 0:
            vis = keypoint_weights[..., 0] > 0
            bone_loss_xy = self._bone_loss(
                pred_ref_xyz[..., :2],
                label_2d,
                vis,
                self.bone_pairs,
                min_gt_len=self.bone_min_gt_len,
                huber_delta=self.bone_huber_delta,
                use_relative=self.bone_use_relative,
                clamp_per_bone=self.bone_clamp_per_bone,
            )
            bone_loss = bone_loss_xy

            if (not self.bone_loss_3d_only) and Bd > 0:
                assert label_depth is not None and label_depth_id is not None
                gt_xyz = torch.cat([torch.index_select(label_2d, 0, label_depth_id), label_depth], dim=-1)
                pred_xyz_d = torch.index_select(pred_ref_xyz, 0, label_depth_id)
                vis_d = torch.index_select(vis, 0, label_depth_id)
                bone_loss_xyz = self._bone_loss(
                    pred_xyz_d,
                    gt_xyz,
                    vis_d,
                    self.bone_pairs,
                    min_gt_len=self.bone_min_gt_len,
                    huber_delta=self.bone_huber_delta,
                    use_relative=self.bone_use_relative,
                    clamp_per_bone=self.bone_clamp_per_bone,
                )
                bone_loss = 0.5 * bone_loss_xy + 0.5 * bone_loss_xyz

            losses["loss_bone"] = bone_loss * self.bone_loss_weight

        # OKS loss
        if self.use_oks_loss and self.oks_loss_weight > 0:
            vis = keypoint_weights[..., 0] > 0
            pred_xy = pred_ref_xyz[..., :2]
            gt_xy = label_2d

            if self.coord_is_normalized:
                if "input_size" in batch_data_samples[0].metainfo:
                    W_in, H_in = batch_data_samples[0].metainfo["input_size"]
                else:
                    W_in, H_in = int(self.input_size[0]), int(self.input_size[1])
                scale_xy = torch.tensor([W_in, H_in], device=device, dtype=pred_xy.dtype).view(1, 1, 2)
                pred_xy_px = pred_xy * scale_xy
                gt_xy_px = gt_xy * scale_xy
            else:
                pred_xy_px = pred_xy
                gt_xy_px = gt_xy

            area = self._get_oks_area(batch_data_samples, gt_xy_px, vis)
            oks = self._oks(pred_xy_px, gt_xy_px, vis, area)
            if self.oks_loss_type == "neglog":
                loss_oks = (-torch.log(oks.clamp(min=self.oks_eps))).mean()
            else:
                loss_oks = (1.0 - oks).mean()
            losses["loss_oks"] = loss_oks * self.oks_loss_weight

        # coarse aux loss
        if self.coarse_loss_weight > 0:
            c_depth_pred, c_depth_label, c_depth_w = _depth_branch(pred_coarse_xyz)
            c_inputs = [pred_coarse_xyz[..., :2], c_depth_pred]
            c_targets = [label_2d, c_depth_label]
            c_weights = [keypoint_weights, c_depth_w]
            c_loss_list = self.loss_module(c_inputs, c_targets, c_weights)
            losses["loss_kpt2d_coarse"] = c_loss_list[0] * self.coarse_loss_weight
            losses["loss_depth_coarse"] = c_loss_list[1] * self.coarse_loss_weight

        _, avg_acc, _ = keypoint_pck_accuracy(
            pred=to_numpy(pred_ref_xyz[..., :2]),
            gt=to_numpy(label_2d),
            mask=to_numpy(keypoint_weights[..., 0]) > 0,
            thr=0.05,
            norm_factor=np.ones((pred_ref_xyz.size(0), 2), dtype=np.float32),
        )
        losses["acc_pose"] = torch.tensor(avg_acc, device=device)
        return losses
