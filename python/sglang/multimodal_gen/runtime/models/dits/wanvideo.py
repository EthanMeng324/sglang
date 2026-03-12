# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import math
import os
import queue
import threading
from functools import lru_cache
from typing import Any

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.dits import WanVideoConfig
from sglang.multimodal_gen.configs.sample.wan import WanTeaCacheParams
from sglang.multimodal_gen.runtime.distributed import (
    divide,
    get_sp_group,
    get_sp_world_size,
    get_tp_world_size,
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.layers.attention import (
    MinimalA2AAttnOp,
    UlyssesAttention_VSA,
    USPAttention,
)
from sglang.multimodal_gen.runtime.layers.elementwise import MulAdd
from sglang.multimodal_gen.runtime.layers.layernorm import (
    FP32LayerNorm,
    LayerNormScaleShift,
    RMSNorm,
    ScaleResidualLayerNormScaleShift,
    tensor_parallel_rms_norm,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.mlp import MLP
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    NDRotaryEmbedding,
    _apply_rotary_emb,
    apply_flashinfer_rope_qk_inplace,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import (
    ModulateProjection,
    PatchEmbed,
    TimestepEmbedder,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
_is_cuda = current_platform.is_cuda()


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


@torch.compiler.disable
def _set_forward_context_comm_tracker(forward_context, tracker) -> None:
    if getattr(forward_context, "comm_activity_tracker", None) is not tracker:
        forward_context.comm_activity_tracker = tracker


class WanImageEmbedding(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = MLP(in_features, in_features, out_features, act_type="gelu")
        self.norm2 = FP32LayerNorm(out_features)

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        dtype = encoder_hidden_states_image.dtype
        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states).to(dtype)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):

    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        text_embed_dim: int,
        image_embed_dim: int | None = None,
    ):
        super().__init__()

        self.time_embedder = TimestepEmbedder(
            dim, frequency_embedding_size=time_freq_dim, act_layer="silu"
        )
        self.time_modulation = ModulateProjection(dim, factor=6, act_layer="silu")
        self.text_embedder = MLP(
            text_embed_dim, dim, dim, bias=True, act_type="gelu_pytorch_tanh"
        )

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        timestep_seq_len: int | None = None,
    ):
        temb = self.time_embedder(timestep, timestep_seq_len)
        timestep_proj = self.time_modulation(temb)

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            assert self.image_embedder is not None
            encoder_hidden_states_image = self.image_embedder(
                encoder_hidden_states_image
            )

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanSelfAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        parallel_attention=False,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ) -> None:
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.parallel_attention = parallel_attention
        tp_size = get_tp_world_size()

        # layers
        self.to_q = ColumnParallelLinear(dim, dim, gather_output=False)
        self.to_k = ColumnParallelLinear(dim, dim, gather_output=False)
        self.to_v = ColumnParallelLinear(dim, dim, gather_output=False)
        self.to_out = RowParallelLinear(dim, dim, input_is_parallel=True)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.tp_rmsnorm = tp_size > 1 and qk_norm
        self.local_num_heads = divide(num_heads, tp_size)

        # Scaled dot product attention
        self.attn = USPAttention(
            num_heads=self.local_num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor, context_lens: int):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        pass


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        q, _ = self.to_q(x)
        if self.tp_rmsnorm:
            q = tensor_parallel_rms_norm(q, self.norm_q)
        else:
            q = self.norm_q(q)
        q = q.unflatten(2, (self.local_num_heads, self.head_dim))

        k, _ = self.to_k(context)
        if self.tp_rmsnorm:
            k = tensor_parallel_rms_norm(k, self.norm_k)
        else:
            k = self.norm_k(k)
        k = k.unflatten(2, (self.local_num_heads, self.head_dim))

        v, _ = self.to_v(context)
        v = v.unflatten(2, (self.local_num_heads, self.head_dim))

        # compute attention
        x = self.attn(q, k, v)

        # output
        x = x.flatten(2)
        x, _ = self.to_out(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ) -> None:
        super().__init__(
            dim,
            num_heads,
            window_size,
            qk_norm,
            eps,
            supported_attention_backends=supported_attention_backends,
        )

        self.add_k_proj = ColumnParallelLinear(dim, dim, gather_output=False)
        self.add_v_proj = ColumnParallelLinear(dim, dim, gather_output=False)
        self.norm_added_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]

        q, _ = self.to_q(x)
        if self.tp_rmsnorm:
            q = tensor_parallel_rms_norm(q, self.norm_q)
        else:
            q = self.norm_q(q)
        q = q.unflatten(2, (self.local_num_heads, self.head_dim))

        k, _ = self.to_k(context)
        if self.tp_rmsnorm:
            k = tensor_parallel_rms_norm(k, self.norm_k)
        else:
            k = self.norm_k(k)
        k = k.unflatten(2, (self.local_num_heads, self.head_dim))

        v, _ = self.to_v(context)
        v = v.unflatten(2, (self.local_num_heads, self.head_dim))

        k_img, _ = self.add_k_proj(context_img)
        if self.tp_rmsnorm:
            k_img = tensor_parallel_rms_norm(k_img, self.norm_added_k)
        else:
            k_img = self.norm_added_k(k_img)
        k_img = k_img.unflatten(2, (self.local_num_heads, self.head_dim))

        v_img, _ = self.add_v_proj(context_img)
        v_img = v_img.unflatten(2, (self.local_num_heads, self.head_dim))

        img_x = self.attn(q, k_img, v_img)
        x = self.attn(q, k, v)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x, _ = self.to_out(x)
        return x


class WanTransformerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
        attention_type: str = "original",
        sla_topk: float = 0.1,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = LayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )
        self.to_q = ColumnParallelLinear(dim, dim, bias=True, gather_output=False)
        self.to_k = ColumnParallelLinear(dim, dim, bias=True, gather_output=False)
        self.to_v = ColumnParallelLinear(dim, dim, bias=True, gather_output=False)

        self.to_out = RowParallelLinear(dim, dim, bias=True, reduce_results=True)
        tp_size = get_tp_world_size()
        self.local_num_heads = divide(num_heads, tp_size)
        self_attn_backends = supported_attention_backends

        if attention_type in ("sla", "sagesla"):
            self.attn1 = MinimalA2AAttnOp(
                num_heads=self.local_num_heads,
                head_size=dim // num_heads,
                attention_type=attention_type,
                topk=sla_topk,
                supported_attention_backends={
                    AttentionBackendEnum.SLA_ATTN,
                    AttentionBackendEnum.SAGE_SLA_ATTN,
                },
            )
        else:
            self.attn1 = USPAttention(
                num_heads=self.local_num_heads,
                head_size=dim // num_heads,
                causal=False,
                supported_attention_backends=self_attn_backends,
                prefix=f"{prefix}.attn1",
            )

        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        self.dim_head = dim // num_heads
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(self.dim_head, eps=eps)
            self.norm_k = RMSNorm(self.dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            logger.error("QK Norm type not supported")
            raise Exception
        assert cross_attn_norm is True
        self.qk_norm = qk_norm
        self.tp_rmsnorm = qk_norm == "rms_norm_across_heads" and tp_size > 1
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=True,
            dtype=torch.float32,
        )

        # 2. Cross-attention
        cross_attn_backends = {
            b for b in supported_attention_backends if not b.is_sparse
        }
        if added_kv_proj_dim is not None:
            # I2V
            self.attn2 = WanI2VCrossAttention(
                dim,
                num_heads,
                qk_norm=qk_norm,
                eps=eps,
                supported_attention_backends=cross_attn_backends,
            )
        else:
            # T2V
            self.attn2 = WanT2VCrossAttention(
                dim,
                num_heads,
                qk_norm=qk_norm,
                eps=eps,
                supported_attention_backends=cross_attn_backends,
            )
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )

        # 3. Feed-forward
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
        self.mlp_residual = MulAdd()

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        mock_comm_fn: Any | None = None,
        block_idx: int | None = None,
        mock_comm_stats: dict[str, int] | None = None,
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        if temb.dim() == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            e = self.scale_shift_table + temb.float()
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                e.chunk(6, dim=1)
            )

        assert shift_msa.dtype == torch.float32

        # 1. Self-attention
        norm_hidden_states = self.norm1(hidden_states, shift_msa, scale_msa)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)

        if self.norm_q is not None:
            if self.tp_rmsnorm:
                query = tensor_parallel_rms_norm(query, self.norm_q)
            else:
                query = self.norm_q(query)
        if self.norm_k is not None:
            if self.tp_rmsnorm:
                key = tensor_parallel_rms_norm(key, self.norm_k)
            else:
                key = self.norm_k(key)
        query = query.squeeze(1).unflatten(2, (self.local_num_heads, self.dim_head))
        key = key.squeeze(1).unflatten(2, (self.local_num_heads, self.dim_head))
        value = value.squeeze(1).unflatten(2, (self.local_num_heads, self.dim_head))

        # Apply rotary embeddings
        cos, sin = freqs_cis
        if _is_cuda and query.shape == key.shape:
            cos_sin_cache = torch.cat(
                [
                    cos.to(dtype=torch.float32).contiguous(),
                    sin.to(dtype=torch.float32).contiguous(),
                ],
                dim=-1,
            )
            query, key = apply_flashinfer_rope_qk_inplace(
                query, key, cos_sin_cache, is_neox=False
            )
        else:
            query, key = _apply_rotary_emb(
                query, cos, sin, is_neox_style=False
            ), _apply_rotary_emb(key, cos, sin, is_neox_style=False)
        if mock_comm_fn is not None and block_idx is not None:
            transferred = int(
                mock_comm_fn(block_idx=block_idx, hidden_states=hidden_states)
            )
            if mock_comm_stats is not None and transferred > 0:
                mock_comm_stats["calls"] = mock_comm_stats.get("calls", 0) + 1
                mock_comm_stats["bytes"] = mock_comm_stats.get("bytes", 0) + transferred

        attn_output = self.attn1(query, key, value)
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = torch.zeros(
            (1,), device=hidden_states.device, dtype=hidden_states.dtype
        )
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale
        )
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype
        ), hidden_states.to(orig_dtype)

        # 2. Cross-attention
        attn_output = self.attn2(
            norm_hidden_states, context=encoder_hidden_states, context_lens=None
        )
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa
        )
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype
        ), hidden_states.to(orig_dtype)

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(ff_output, c_gate_msa, hidden_states)
        hidden_states = hidden_states.to(orig_dtype)

        return hidden_states


class WanTransformerBlock_VSA(nn.Module):

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = LayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )
        self.to_q = ColumnParallelLinear(dim, dim, bias=True, gather_output=True)
        self.to_k = ColumnParallelLinear(dim, dim, bias=True, gather_output=True)
        self.to_v = ColumnParallelLinear(dim, dim, bias=True, gather_output=True)
        self.to_gate_compress = ColumnParallelLinear(
            dim, dim, bias=True, gather_output=True
        )

        self.to_out = ColumnParallelLinear(dim, dim, bias=True, gather_output=True)
        self.attn1 = UlyssesAttention_VSA(
            num_heads=num_heads,
            head_size=dim // num_heads,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn1",
        )
        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        dim_head = dim // num_heads
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            logger.error("QK Norm type not supported")
            raise Exception
        assert cross_attn_norm is True
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=True,
            dtype=torch.float32,
        )

        # 2. Cross-attention
        cross_attn_backends = {
            b for b in supported_attention_backends if not b.is_sparse
        }
        if added_kv_proj_dim is not None:
            # I2V
            self.attn2 = WanI2VCrossAttention(
                dim,
                num_heads,
                qk_norm=qk_norm,
                eps=eps,
                supported_attention_backends=cross_attn_backends,
            )
        else:
            # T2V
            self.attn2 = WanT2VCrossAttention(
                dim,
                num_heads,
                qk_norm=qk_norm,
                eps=eps,
                supported_attention_backends=cross_attn_backends,
            )
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )

        # 3. Feed-forward
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
        self.mlp_residual = MulAdd()

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        mock_comm_fn: Any | None = None,
        block_idx: int | None = None,
        mock_comm_stats: dict[str, int] | None = None,
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        # assert orig_dtype != torch.float32
        e = self.scale_shift_table + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(
            6, dim=1
        )
        assert shift_msa.dtype == torch.float32

        # 1. Self-attention
        norm_hidden_states = self.norm1(hidden_states, shift_msa, scale_msa)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)
        gate_compress, _ = self.to_gate_compress(norm_hidden_states)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        gate_compress = gate_compress.squeeze(1).unflatten(
            2, (self.num_attention_heads, -1)
        )

        # Apply rotary embeddings
        cos, sin = freqs_cis
        if _is_cuda and query.shape == key.shape:
            cos_sin_cache = torch.cat(
                [
                    cos.to(dtype=torch.float32).contiguous(),
                    sin.to(dtype=torch.float32).contiguous(),
                ],
                dim=-1,
            )
            query, key = apply_flashinfer_rope_qk_inplace(
                query, key, cos_sin_cache, is_neox=False
            )
        else:
            query, key = _apply_rotary_emb(
                query, cos, sin, is_neox_style=False
            ), _apply_rotary_emb(key, cos, sin, is_neox_style=False)

        if mock_comm_fn is not None and block_idx is not None:
            transferred = int(
                mock_comm_fn(block_idx=block_idx, hidden_states=hidden_states)
            )
            if mock_comm_stats is not None and transferred > 0:
                mock_comm_stats["calls"] = mock_comm_stats.get("calls", 0) + 1
                mock_comm_stats["bytes"] = mock_comm_stats.get("bytes", 0) + transferred

        attn_output = self.attn1(query, key, value, gate_compress=gate_compress)
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = torch.zeros((1,), device=hidden_states.device)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale
        )
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype
        ), hidden_states.to(orig_dtype)

        # 2. Cross-attention
        attn_output = self.attn2(
            norm_hidden_states, context=encoder_hidden_states, context_lens=None
        )
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa
        )
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype
        ), hidden_states.to(orig_dtype)

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(ff_output, c_gate_msa, hidden_states)
        hidden_states = hidden_states.to(orig_dtype)

        return hidden_states


class WanTransformer3DModel(CachableDiT, OffloadableDiTMixin):
    _fsdp_shard_conditions = WanVideoConfig()._fsdp_shard_conditions
    _compile_conditions = WanVideoConfig()._compile_conditions
    _supported_attention_backends = WanVideoConfig()._supported_attention_backends
    param_names_mapping = WanVideoConfig().param_names_mapping
    reverse_param_names_mapping = WanVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanVideoConfig().lora_param_names_mapping

    def __init__(self, config: WanVideoConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size
        self.text_len = config.text_len

        # 1. Patch & position embedding
        self.patch_embedding = PatchEmbed(
            in_chans=config.in_channels,
            embed_dim=inner_dim,
            patch_size=config.patch_size,
            flatten=False,
        )

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
            image_embed_dim=config.image_dim,
        )

        # 3. Transformer blocks
        attn_backend = get_global_server_args().attention_backend
        transformer_block = (
            WanTransformerBlock_VSA
            if (attn_backend and attn_backend.lower() == "video_sparse_attn")
            else WanTransformerBlock
        )
        self.blocks = nn.ModuleList(
            [
                transformer_block(
                    inner_dim,
                    config.ffn_dim,
                    config.num_attention_heads,
                    config.qk_norm,
                    config.cross_attn_norm,
                    config.eps,
                    config.added_kv_proj_dim,
                    self._supported_attention_backends
                    | {AttentionBackendEnum.VIDEO_SPARSE_ATTN},
                    prefix=f"{config.prefix}.blocks.{i}",
                    attention_type=config.attention_type,
                    sla_topk=config.sla_topk,
                )
                for i in range(config.num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = LayerNormScaleShift(
            inner_dim,
            eps=config.eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )
        self.proj_out = ColumnParallelLinear(
            inner_dim,
            config.out_channels * math.prod(config.patch_size),
            bias=True,
            gather_output=True,
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        # For type checking

        self.cnt = 0
        self.__post_init__()

        # misc
        self.sp_size = get_sp_world_size()

        # Get rotary embeddings
        d = self.hidden_size // self.num_attention_heads
        self.rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]

        self.rotary_emb = NDRotaryEmbedding(
            rope_dim_list=self.rope_dim_list,
            rope_theta=10000,
            dtype=(
                torch.float32
                if current_platform.is_mps() or current_platform.is_musa()
                else torch.float64
            ),
        )

        self.layer_names = ["blocks"]
        self._mock_comm_enabled = _env_bool("SGLANG_WAN_MOCK_COMM_ENABLE", False)
        self._mock_comm_every_n_blocks = max(
            1, int(os.getenv("SGLANG_WAN_MOCK_COMM_EVERY_N_BLOCKS", "1"))
        )
        self._mock_comm_debug = _env_bool("SGLANG_WAN_MOCK_COMM_DEBUG", False)
        self._mock_comm_fixed_mb = max(
            0.0, _env_float("SGLANG_WAN_MOCK_COMM_PCIE_MB", 64.0)
        )
        self._mock_comm_virtual_sp_degree = max(
            2, _env_int("SGLANG_WAN_MOCK_COMM_VIRTUAL_SP_DEGREE", 2)
        )
        self._mock_comm_traffic_scale = max(
            0.01, _env_float("SGLANG_WAN_MOCK_COMM_TRAFFIC_SCALE", 1.0)
        )
        self._mock_comm_max_mb = max(
            1, _env_int("SGLANG_WAN_MOCK_COMM_MAX_MB", 256)
        )

        self._mock_comm_capacity_bytes = 0
        self._mock_comm_gpu_buffer: torch.Tensor | None = None
        self._mock_comm_cpu_buffer: torch.Tensor | None = None
        self._mock_comm_event_q: (
            queue.Queue[tuple[torch.cuda.Event, torch.cuda.Event, str, bool]] | None
        ) = None
        self._mock_comm_event_stop = threading.Event()
        self._mock_comm_event_thread: threading.Thread | None = None

        if self._mock_comm_enabled:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "SGLANG_WAN_MOCK_COMM_ENABLE=1 requires CUDA, but CUDA is unavailable."
                )
            self._mock_comm_event_q = queue.Queue()
            self._mock_comm_event_thread = threading.Thread(
                target=self._mock_comm_event_loop,
                name="WanMockCommEventLoop",
                daemon=True,
            )
            self._mock_comm_event_thread.start()

        if self._mock_comm_enabled:
            logger.info(
                "Wan mock communication (PCIe transfer) is enabled "
                f"(every_n_blocks={self._mock_comm_every_n_blocks}, "
                f"fixed_mb={self._mock_comm_fixed_mb}, "
                f"virtual_sp_degree={self._mock_comm_virtual_sp_degree}, "
                f"traffic_scale={self._mock_comm_traffic_scale}, "
                f"max_mb={self._mock_comm_max_mb})."
            )

    def _mock_comm_event_loop(self) -> None:
        if self._mock_comm_event_q is None:
            return
        pending: list[tuple[torch.cuda.Event, torch.cuda.Event, str, bool, bool]] = []
        while not self._mock_comm_event_stop.is_set():
            made_progress = False

            # Drain queued mock-comm event pairs.
            while True:
                try:
                    item = self._mock_comm_event_q.get_nowait()
                except queue.Empty:
                    break

                # Backward-compatible tuple parsing:
                # old/new queue payload: (start_event, end_event, tag)
                # transient payload from intermediate revisions:
                # (start_event, end_event, tag, tracker_started_inline)
                if len(item) == 3:
                    start_event, end_event, tag = item  # type: ignore[misc]
                    started_inline = False
                else:
                    start_event, end_event, tag, started_inline = item  # type: ignore[misc]
                pending.append(
                    (start_event, end_event, tag, bool(started_inline), False)
                )
                made_progress = True

            next_pending: list[
                tuple[torch.cuda.Event, torch.cuda.Event, str, bool, bool]
            ] = []
            tracker = getattr(self, "comm_activity_tracker", None)
            for start_event, end_event, tag, started, active_nvtx_pushed in pending:
                if not started and start_event.query():
                    if torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"):
                        torch.cuda.nvtx.range_push("SGL_MOCK_COMM_ACTIVE")
                        active_nvtx_pushed = True
                    if tracker is not None:
                        tracker.mark_start(tag)
                    started = True
                    made_progress = True

                if started and end_event.query():
                    if tracker is not None:
                        tracker.mark_end(tag)
                    if (
                        active_nvtx_pushed
                        and torch.cuda.is_available()
                        and hasattr(torch.cuda, "nvtx")
                    ):
                        torch.cuda.nvtx.range_pop()
                    made_progress = True
                    continue

                next_pending.append(
                    (start_event, end_event, tag, started, active_nvtx_pushed)
                )
            pending = next_pending

            if made_progress:
                continue

            try:
                item = self._mock_comm_event_q.get(timeout=0.001)
            except queue.Empty:
                continue
            if len(item) == 3:
                start_event, end_event, tag = item  # type: ignore[misc]
                started_inline = False
            else:
                start_event, end_event, tag, started_inline = item  # type: ignore[misc]
            pending.append(
                (start_event, end_event, tag, bool(started_inline), False)
            )

    def _estimate_mock_comm_bytes(self, hidden_states: torch.Tensor) -> int:
        if self._mock_comm_fixed_mb > 0:
            return int(self._mock_comm_fixed_mb * 1024 * 1024)

        hidden_bytes = hidden_states.numel() * hidden_states.element_size()
        one_way = hidden_bytes * (
            self._mock_comm_virtual_sp_degree - 1
        ) // self._mock_comm_virtual_sp_degree
        estimated = int(2 * one_way * self._mock_comm_traffic_scale)
        max_bytes = self._mock_comm_max_mb * 1024 * 1024
        return max(1, min(estimated, max_bytes))

    def _ensure_mock_comm_buffers(self, num_bytes: int) -> None:
        if (
            self._mock_comm_gpu_buffer is not None
            and self._mock_comm_cpu_buffer is not None
            and self._mock_comm_capacity_bytes >= num_bytes
        ):
            return

        self._mock_comm_gpu_buffer = torch.empty(
            num_bytes, device=torch.cuda.current_device(), dtype=torch.uint8
        )
        self._mock_comm_cpu_buffer = torch.empty(
            num_bytes, device="cpu", dtype=torch.uint8, pin_memory=True
        )
        self._mock_comm_capacity_bytes = num_bytes

    @torch.compiler.disable
    def _maybe_mock_communication(
        self, *, block_idx: int, hidden_states: torch.Tensor
    ) -> int:
        if not self._mock_comm_enabled:
            return 0
        if (block_idx % self._mock_comm_every_n_blocks) != 0:
            return 0
        tag = f"wan_mock_comm_block_{block_idx}"
        tracker = getattr(self, "comm_activity_tracker", None)
        if tracker is not None and self._mock_comm_event_q is None:
            raise RuntimeError(
                "Mock communication is enabled but GPU mock-comm runtime is not initialized."
            )

        num_bytes = self._estimate_mock_comm_bytes(hidden_states)
        # logger.info(
        #     f"Wan mock communication: num_bytes={num_bytes}"
        # )
        self._ensure_mock_comm_buffers(num_bytes)
        assert self._mock_comm_gpu_buffer is not None
        assert self._mock_comm_cpu_buffer is not None

        # One round-trip on PCIe: D2H then H2D. This is on critical path.
        start_event = torch.cuda.Event()
        start_event.record(torch.cuda.current_stream())
        tracker_started_inline = False
        if tracker is not None:
            # Strict mode: block prefetch immediately when mock comm is scheduled,
            # then release only after end_event is observed by the watcher thread.
            tracker.mark_start(tag)
            tracker_started_inline = True
            quiesce_fn = getattr(self, "quiesce_prefetch_for_comm", None)
            if callable(quiesce_fn):
                quiesce_fn()
        if torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"):
            torch.cuda.nvtx.range_push("SGL_MOCK_COMM_PCIE")
        try:
            self._mock_comm_cpu_buffer[:num_bytes].copy_(
                self._mock_comm_gpu_buffer[:num_bytes], non_blocking=True
            )
            # self._mock_comm_gpu_buffer[:num_bytes].copy_(
            #     self._mock_comm_cpu_buffer[:num_bytes], non_blocking=True
            # )
        finally:
            if torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"):
                torch.cuda.nvtx.range_pop()
        end_event = torch.cuda.Event()
        end_event.record(torch.cuda.current_stream())
        if self._mock_comm_event_q is not None:
            self._mock_comm_event_q.put(
                (start_event, end_event, tag, tracker_started_inline)
            )
        return num_bytes

    @lru_cache(maxsize=1)
    def _compute_rope_for_sequence_shard(
        self,
        local_len: int,
        rank: int,
        frame_stride_local: int,
        width_local: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_start = rank * local_len
        token_indices = torch.arange(
            token_start,
            token_start + local_len,
            device=device,
            dtype=torch.long,
        )
        t_idx = token_indices // frame_stride_local
        rem = token_indices % frame_stride_local
        h_idx = rem // width_local
        w_idx = rem % width_local
        positions = torch.stack((t_idx, h_idx, w_idx), dim=1)
        return self.rotary_emb.forward_uncached(positions)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance=None,
        **kwargs,
    ) -> torch.Tensor:
        forward_context = get_forward_context()
        tracker = getattr(self, "comm_activity_tracker", None)
        _set_forward_context_comm_tracker(forward_context, tracker)

        forward_batch = forward_context.forward_batch
        if forward_batch is not None:
            sequence_shard_enabled = (
                forward_batch.enable_sequence_shard and self.sp_size > 1
            )
        else:
            sequence_shard_enabled = False
        self.enable_teacache = (
            forward_batch is not None and forward_batch.enable_teacache
        )

        orig_dtype = hidden_states.dtype
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if (
            isinstance(encoder_hidden_states_image, list)
            and len(encoder_hidden_states_image) > 0
        ):
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        else:
            encoder_hidden_states_image = None

        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        if not sequence_shard_enabled:
            # The rotary embedding layer correctly handles SP offsets internally.
            freqs_cos, freqs_sin = self.rotary_emb.forward_from_grid(
                (
                    post_patch_num_frames * self.sp_size,
                    post_patch_height,
                    post_patch_width,
                ),
                shard_dim=0,
                start_frame=0,
                device=hidden_states.device,
            )
            assert freqs_cos.dtype == torch.float32
            assert freqs_cos.device == hidden_states.device
            freqs_cis = (
                (freqs_cos.float(), freqs_sin.float())
                if freqs_cos is not None
                else None
            )

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # shape is [B, T' * H' * W', C]
        seq_len_orig = hidden_states.shape[1]
        seq_shard_pad = 0
        if sequence_shard_enabled:
            if seq_len_orig % self.sp_size != 0:
                seq_shard_pad = self.sp_size - (seq_len_orig % self.sp_size)
                pad = torch.zeros(
                    (batch_size, seq_shard_pad, hidden_states.shape[2]),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                hidden_states = torch.cat([hidden_states, pad], dim=1)
            sp_rank = get_sp_group().rank_in_group
            local_seq_len = hidden_states.shape[1] // self.sp_size
            hidden_states = hidden_states.view(
                batch_size, self.sp_size, local_seq_len, hidden_states.shape[2]
            )
            hidden_states = hidden_states[:, sp_rank, :, :]

            frame_stride = post_patch_height * post_patch_width
            freqs_cos, freqs_sin = self._compute_rope_for_sequence_shard(
                local_seq_len,
                sp_rank,
                frame_stride,
                post_patch_width,
                hidden_states.device,
            )
            freqs_cis = (freqs_cos.float(), freqs_sin.float())

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.dim() == 2:
            # ti2v
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
                timestep_seq_len=ts_seq_len,
            )
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )

        encoder_hidden_states = (
            encoder_hidden_states.to(orig_dtype)
            if not current_platform.is_amp_supported()
            else encoder_hidden_states
        )  # cast to orig_dtype for MPS

        assert encoder_hidden_states.dtype == orig_dtype

        # 4. Transformer blocks
        # if caching is enabled, we might be able to skip the forward pass
        should_skip_forward = self.should_skip_forward_for_cached_states(
            timestep_proj=timestep_proj, temb=temb
        )

        if should_skip_forward:
            hidden_states = self.retrieve_cached_states(hidden_states)
        else:
            # if teacache is enabled, we need to cache the original hidden states
            if self.enable_teacache:
                original_hidden_states = hidden_states.clone()

            mock_comm_stats = {"calls": 0, "bytes": 0}
            for block_idx, block in enumerate(self.blocks):
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    freqs_cis,
                    mock_comm_fn=self._maybe_mock_communication,
                    block_idx=block_idx,
                    mock_comm_stats=mock_comm_stats,
                )
            if self._mock_comm_debug and mock_comm_stats["calls"] > 0:
                logger.info(
                    "Wan mock communication injected in this forward: "
                    f"calls={mock_comm_stats['calls']}, "
                    f"total_transfer_mb={mock_comm_stats['bytes'] / (1024**2):.2f}, "
                    f"per_call_mb={mock_comm_stats['bytes'] / (mock_comm_stats['calls'] * 1024**2):.2f}"
                )
            # if teacache is enabled, we need to cache the original hidden states
            if self.enable_teacache:
                self.maybe_cache_states(hidden_states, original_hidden_states)
        self.cnt += 1

        if sequence_shard_enabled:
            hidden_states = hidden_states.contiguous()
            hidden_states = sequence_model_parallel_all_gather(hidden_states, dim=1)
            if seq_shard_pad > 0:
                hidden_states = hidden_states[:, :seq_len_orig, :]

        # 5. Output norm, projection & unpatchify
        if temb.dim() == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (
                self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)
            ).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states, _ = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return output

    def maybe_cache_states(
        self, hidden_states: torch.Tensor, original_hidden_states: torch.Tensor
    ) -> None:
        """Cache residual with CFG positive/negative separation."""
        residual = hidden_states.squeeze(0) - original_hidden_states
        if not self.is_cfg_negative:
            self.previous_residual = residual
        else:
            self.previous_residual_negative = residual

    def should_skip_forward_for_cached_states(self, **kwargs) -> bool:
        if not self.enable_teacache:
            return False
        ctx = self._get_teacache_context()
        if ctx is None:
            return False

        # Wan uses WanTeaCacheParams with additional fields
        teacache_params = ctx.teacache_params
        assert isinstance(
            teacache_params, WanTeaCacheParams
        ), "teacache_params is not a WanTeaCacheParams"

        # Initialize Wan-specific parameters
        use_ret_steps = teacache_params.use_ret_steps
        cutoff_steps = teacache_params.get_cutoff_steps(ctx.num_inference_steps)
        ret_steps = teacache_params.ret_steps

        # Adjust ret_steps and cutoff_steps for non-CFG mode
        # (WanTeaCacheParams uses *2 factor assuming CFG)
        if not ctx.do_cfg:
            ret_steps = ret_steps // 2
            cutoff_steps = cutoff_steps // 2

        timestep_proj = kwargs["timestep_proj"]
        temb = kwargs["temb"]
        modulated_inp = timestep_proj if use_ret_steps else temb

        self.is_cfg_negative = ctx.is_cfg_negative

        # Wan uses ret_steps/cutoff_steps for boundary detection
        is_boundary_step = self.cnt < ret_steps or self.cnt >= cutoff_steps

        # Use shared helper to compute cache decision
        should_calc = self._compute_teacache_decision(
            modulated_inp=modulated_inp,
            is_boundary_step=is_boundary_step,
            coefficients=ctx.coefficients,
            teacache_thresh=ctx.teacache_thresh,
        )

        return not should_calc

    def retrieve_cached_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Retrieve cached residual with CFG positive/negative separation."""
        if not self.is_cfg_negative:
            return hidden_states + self.previous_residual
        else:
            return hidden_states + self.previous_residual_negative


EntryClass = WanTransformer3DModel
