from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_forward, benchmark_all, pytorch_profiler
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
# from flash_attn.triton.fused_attention import attention as attention
from flash_attn.flash_attn_triton import flash_attn_qkvpacked_func
from flash_attn.flash_attn_triton_og import attention as attention_og

try:
    from flash_attn.fused_softmax import scaled_upper_triang_masked_softmax
except ImportError:
    scaled_upper_triang_masked_softmax = None


def attention_pytorch(qkv, dropout_p=0.0, causal=True):
    """
    Compute self-attention using PyTorch operations.

    Args:
        qkv: (batch_size, seqlen, 3, nheads, head_dim) - The input tensor containing queries, keys, and values.
        dropout_p: float - The probability of an element to be zeroed in dropout.
        causal: bool - Whether to use a causal mask for attention.

    Returns:
        output: (batch_size, seqlen, nheads, head_dim) - The output tensor after self-attention.
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, 'b t h d -> (b h) t d')  # Reshape queries tensor
    k = rearrange(k, 'b s h d -> (b h) d s')  # Reshape keys tensor
    softmax_scale = 1.0 / math.sqrt(d)

    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale), '(b h) t s -> b h t s', h=nheads)

    if causal:
        # "triu_tril_cuda_template" not implemented for 'BFloat16'
        # So we have to construct the mask in float
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
        scores = scores + causal_mask.to(dtype=scores.dtype)

    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop, v)  # Compute output using einsum

    return output.to(dtype=qkv.dtype)

def attention_megatron(qkv):
    """
    Compute self-attention using the Megatron-style implementation.

    Args:
        qkv: (batch_size, seqlen, 3, nheads, head_dim) - The input tensor containing queries, keys, and values.

    Returns:
        output: (batch_size, seqlen, nheads, head_dim) - The output tensor after self-attention.
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)  # Unbind the input tensor along the third dimension to get separate tensors for queries, keys, and values
    q = rearrange(q, 'b t h d -> (b h) t d')  # Reshape the queries tensor
    k = rearrange(k, 'b s h d -> (b h) d s')  # Reshape the keys tensor
    softmax_scale = 1.0 / math.sqrt(d)  # Compute the softmax scale

    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=nheads)  # Compute the attention scores

    # Apply scaled upper triangular masked softmax to the scores
    attention = scaled_upper_triang_masked_softmax(scores, None, scale=1.0)

    # Compute the output using matrix multiplication and einsum
    output = torch.einsum('bhts,bshd->bthd', attention, v)

    return output.to(dtype=qkv.dtype)


# Set random seed for reproducibility
torch.manual_seed(0)

# Define input dimensions and configuration
repeats = 30
batch_size = 2
seqlen = 4096
nheads = 12
headdim = 128
dropout_p = 0.0
causal = True
dtype = torch.bfloat16
device = 'cuda'

# Generate random input tensor qkv
qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype,
                  requires_grad=True)

# Generate cu_seqlens tensor for benchmarking
cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                          device=qkv.device)

# Benchmark the FlashAttention implementation
benchmark_all(flash_attn_unpadded_qkvpacked_func, rearrange(qkv, 'b s ... -> (b s) ...'),
              cu_seqlens, seqlen, dropout_p, causal=causal, repeats=repeats, desc='FlashAttention')

# Benchmark the PyTorch Attention implementation
benchmark_all(attention_pytorch, qkv, dropout_p, causal=causal,
              repeats=repeats, desc='PyTorch Attention')

# Benchmark the FlashAttention Triton implementation
benchmark_all(flash_attn_qkvpacked_func, qkv, None, causal, repeats=repeats, desc='FlashAttention Triton')

# Profile the FlashAttention Triton implementation using PyTorch profiler
pytorch_profiler(flash_attn_qkvpacked_func, qkv, None, causal, backward=True)

# Generate random input tensors q, k, and v
q, k, v = [torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype,
                       requires_grad=True) for _ in range(3)]

# Benchmark the FlashAttention Triton OG implementation
benchmark_all(attention_og, q, k, v, 1.0, repeats=repeats, desc='FlashAttention Triton OG')

# Benchmark the Megatron Attention implementation if scaled_upper_triang_masked_softmax is available
if scaled_upper_triang_masked_softmax is not None:
    benchmark_all(attention_megatron, qkv, repeats=repeats, desc='Megatron Attention')
