"""
From x-transformers: https://github.com/lucidrains/x-transformers
"""

from __future__ import annotations
from typing import Callable, Tuple

import math
from random import random, randrange
from packaging import version

import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList, ModuleDict
from torch.amp import autocast

from functools import partial, wraps
from dataclasses import dataclass

from einops import rearrange, repeat, reduce, pack, unpack

DEFAULT_DIM_HEAD = 64

@dataclass
class Intermediates:
    qk_similarities:    Tensor | None = None
    pre_softmax_attn:   Tensor | None = None
    post_softmax_attn:  Tensor | None = None
    cached_kv:          Tuple[Tensor, Tensor] | None = None
    layer_type:         str | None = None

    def to_tuple(self):
        return (self.qk_similarities, self.pre_softmax_attn, self.post_softmax_attn)

@dataclass
class LayerIntermediates:
    hiddens:            list[Tensor] | None = None   # all hiddens, before the final norm (in pre-norm architecture)
    last_hidden:        Tensor | None = None         # very last hidden after all attention layers, after the final norm
    attn_intermediates: list[Intermediates] | None = None
    layer_hiddens:      list[Tensor] | None = None
    attn_z_loss:        Tensor | None = None
    mems:               Tensor | None = None
    memory_tokens:      Tensor | None = None

LinearNoBias = partial(nn.Linear, bias = False)

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def first(it):
    return it[0]

def is_empty(x):
    return len(x) == 0

def cast_tuple(val, depth = 1):
    return val if isinstance(val, tuple) else (val,) * depth

def divisible_by(num, den):
    return (num % den) == 0

def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

def at_most_one_of(*bools):
    return sum([*map(int, bools)]) <= 1

def compact(arr):
    return [*filter(exists, arr)]

@torch.jit.script
def softclamp(t: Tensor, value: float):
    return (t / value).tanh() * value

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

def selective_attn(
    sim,
    sim_head_gate = None,
    no_mask_sos = True
):
    i, j, device = *sim.shape[-2:], sim.device
    sim_head_gate = default(sim_head_gate, sim[:, 0])

    gate = F.relu(sim_head_gate) # only positive

    if no_mask_sos:
        gate = gate.clone()
        gate[..., -i] = 0.

    eye = torch.eye(i, device = device)

    if j > i:
        eye = F.pad(eye, (j - i, 0), value = 1.)

    gate = (1. - eye) * gate
    gate = F.pad(gate, (0, 0, 1, -1), value = 0.) # only allow for masking the future
    gate = gate.cumsum(dim = -2)

    return sim - rearrange(gate, 'b i j -> b 1 i j')

# alternative distance functions

def qk_l2_dist_squared(q, k):
    if k.ndim == 3:
        k = repeat(k, 'b j d -> b h j d', h = q.shape[1])

    q, packed_shape = pack_one(q, '* i d')
    k, _ = pack_one(k, '* j d')

    l2_dist_squared = torch.cdist(q, k) ** 2
    return unpack_one(l2_dist_squared, packed_shape, '* i j')

# one-hot straight through softmax

def one_hot_straight_through(t, temperature = 1.):
    one_hot_indices = t.argmax(dim = -1, keepdim = True)
    one_hot = torch.zeros_like(t).scatter(-1, one_hot_indices, 1.)

    t = (t / temperature).softmax(dim = -1)
    return one_hot + t - t.detach()

# functions for creating causal mask
# need a special one for onnx cpu (no support for .triu)

def create_causal_mask(i, j, device):
    return torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)

def onnx_create_causal_mask(i, j, device):
    r = torch.arange(i, device = device)
    causal_mask = rearrange(r, 'i -> i 1') < rearrange(r, 'j -> 1 j')
    causal_mask = F.pad(causal_mask, (j - i, 0), value = False)
    return causal_mask

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

def at_most_one_of(*bools):
    return sum(map(int, bools)) <= 1

class always():
    def __init__(self, val):
        self.val = val
    def __call__(self, *args, **kwargs):
        return self.val

class not_equals():
    def __init__(self, val):
        self.val = val
    def __call__(self, x, *args, **kwargs):
        return x != self.val

class equals():
    def __init__(self, val):
        self.val = val
    def __call__(self, x, *args, **kwargs):
        return x == self.val

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# tensor helpers

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def l2norm(t, groups = 1):
    t = rearrange(t, '... (g d) -> ... g d', g = groups)
    t = F.normalize(t, p = 2, dim = -1)
    return rearrange(t, '... g d -> ... (g d)')

def softclamp(t, value):
    return (t / value).tanh() * value

def masked_mean(t, mask = None, dim = 1):
    if not exists(mask):
        return t.mean(dim = dim)

    dims_append = (1,) * (t.ndim - mask.ndim)
    mask = mask.reshape(*mask.shape, *dims_append)

    num = (t * mask).sum(dim = dim)
    den = mask.sum(dim = dim).clamp(min = 1.)
    return num / den

def pad_at_dim(t, pad: tuple[int, int], dim = -1, value = 0.):
    if pad == (0, 0):
        return t

    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head

# auxiliary loss helpers

def calc_z_loss(
    pre_softmax_attns: list[Tensor],
    mask = None,
    weight = 1.
):
    # the same loss applied to the mixture of experts router logits in https://arxiv.org/abs/2202.08906
    # in the paper, in a tiny footnote, they mention using it on attention logits with stabilizing effects
    # also used in PaLM as one of the measures

    lse = 0.

    for attn in pre_softmax_attns:
        lse = lse + attn.logsumexp(dim = -1)

    loss = torch.square(lse)
    loss = reduce(loss, 'b h n -> b n', 'sum')

    if not exists(mask):
        return loss.mean() * weight

    loss = loss[mask].sum() / mask.sum().clamp(min = 1e-5)
    return loss * weight

# init helpers

def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

# keyword argument helpers

def pick_and_pop(keys, d):
    values = tuple(d.pop(key) for key in  keys)
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return tuple(return_val)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    prefix_len = len(prefix)
    kwargs_without_prefix = {key[prefix_len:]: value for key, value in kwargs_with_prefix.items()}
    return kwargs_without_prefix, kwargs

# structured dropout, more effective than traditional attention dropouts

def dropout_seq(seq, mask, dropout):
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device = device)

    if exists(mask):
        mask_value = max_neg_value(logits)
        logits = logits.masked_fill(~mask, mask_value)

    keep_prob = 1. - dropout
    num_keep = max(1,  int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim = 1).indices

    batch_indices = torch.arange(b, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim = -1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device = device) < rearrange(seq_keep_counts, 'b -> b 1')

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask

# activations

class ReluSquared(Module):
    def forward(self, x):
        return F.relu(x) ** 2

# embedding

class TokenEmbedding(Module):
    def __init__(self, dim, num_tokens, l2norm_embed = False):
        super().__init__()
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(num_tokens, dim)

    def forward(self, x):
        token_emb = self.emb(x.long())
        return l2norm(token_emb) if self.l2norm_embed else token_emb

# positional embeddings

class AbsolutePositionalEmbedding(Module):
    def __init__(self, dim, max_seq_len, l2norm_embed = False):
        super().__init__()
        self.scale = dim ** -0.5 if not l2norm_embed else 1.
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if not exists(pos):
            pos = torch.arange(seq_len, device = device)

        if exists(seq_start_pos):
            pos = (pos - seq_start_pos[..., None]).clamp(min = 0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb

class ScaledSinusoidalEmbedding(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        assert divisible_by(dim, 2)
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device

        if not exists(pos):
            pos = torch.arange(seq_len, device = device)

        if exists(seq_start_pos):
            pos = pos - seq_start_pos[..., None]

        emb = einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb * self.scale

class RelativePositionBias(Module):
    def __init__(self, scale, causal = False, num_buckets = 32, max_distance = 128, heads = 8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal = True, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        device = self.device
        q_pos = torch.arange(j - i, j, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> h i j')
        return bias * self.scale

class CoPE(Module):
    """
    Appendix B of https://arxiv.org/abs/2405.18719
    """
    def __init__ (
        self,
        dim,
        heads,
        max_pos,
        soft_onehot = False,
        talking_heads = False,
        soft_onehot_temp = 5e-2
    ):
        super () . __init__ ()
        self.max_pos = max_pos
        self.pos_emb = nn.Parameter(torch.zeros(max_pos, dim))

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else None
        self.soft_onehot = soft_onehot
        self.soft_onehot_temp = soft_onehot_temp

        if soft_onehot:
            self.register_buffer('positions', torch.arange(max_pos))

    def forward(self, query, attn_logits):

        if exists(self.talking_heads):
            i, j = attn_logits.shape[-2:]
            causal_mask = attn_logits.new_ones(i, j).triu_(j - i + 1).bool()

            attn_logits = self.talking_heads(attn_logits)

            attn_logits = attn_logits.masked_fill(causal_mask, -torch.finfo(attn_logits.dtype).max)

        # compute positions

        gates = attn_logits.sigmoid()

        pos = gates.flip(-1).cumsum(dim = -1).flip(-1)
        pos = pos.clamp(max = self.max_pos - 1)

        logits_int = einsum('b h n d, p d -> b h n p', query, self.pos_emb)

        if self.soft_onehot:
            diff_pos = (pos[..., None] - self.positions).abs()
            soft_onehot_pos = F.softmax(-diff_pos / self.soft_onehot_temp, dim = -1)
            cope_pos_emb = einsum('b h i j p, b h i p -> b h i j', soft_onehot_pos, logits_int)
        else:
            # interpolate from integer positions
            pos_ceil = pos.ceil().long()
            pos_floor = pos.floor().long()
            logits_ceil = logits_int.gather(-1, pos_ceil)
            logits_floor = logits_int.gather(-1, pos_floor)

            w = pos - pos_floor
            cope_pos_emb = logits_ceil * w + logits_floor * (1 - w)

        return cope_pos_emb

class DynamicPositionBias(Module):
    def __init__(self, dim, *, heads, depth, log_distance = False, norm = False):
        super().__init__()
        assert depth >= 1, 'depth for dynamic position bias MLP must be greater or equal to 1'
        self.log_distance = log_distance

        self.mlp = ModuleList([])

        self.mlp.append(Sequential(
            nn.Linear(1, dim),
            LayerNorm(dim) if norm else None,
            nn.SiLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim) if norm else None,
                nn.SiLU()
            ))

        self.mlp.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        assert i == j
        n, device = j, self.device

        # get the (n x n) matrix of distances
        seq_arange = torch.arange(n, device = device)
        context_arange = torch.arange(n, device = device)
        indices = rearrange(seq_arange, 'i -> i 1') - rearrange(context_arange, 'j -> 1 j')
        indices += (n - 1)

        # input to continuous positions MLP
        pos = torch.arange(-n + 1, n, device = device).float()
        pos = rearrange(pos, '... -> ... 1')

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            pos = layer(pos)

        # get position biases        
        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')
        return bias

class AlibiPositionalBias(Module):
    def __init__(self, heads, total_heads, **kwargs):
        super().__init__()
        self.heads = heads
        self.total_heads = total_heads

        slopes = Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)
    
    def get_bias(self, i, j, device):
        i_arange = torch.arange(j - i, j, device = device)
        j_arange = torch.arange(j, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, i, j):
        h, device = self.total_heads, self.device

        if exists(self.bias) and self.bias.shape[-1] >= j and self.bias.shape[-2] >= i:
            return self.bias[..., -i:, -j:]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim = 0)
        self.register_buffer('bias', bias, persistent = False)

        return self.bias

class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = torch.arange(seq_len, device = device)
        return self.forward(t)

    @autocast('cuda', enabled = False)
    def forward(self, t):
        max_pos = t.max() + 1

        freqs = torch.einsum('i , j -> i j', t.type_as(self.inv_freq), self.inv_freq) / self.interpolation_factor
        freqs = torch.stack((freqs, freqs), dim = -1)
        freqs = rearrange(freqs, '... d r -> ... (d r)')

        if not exists(self.scale):
            return freqs, 1.

        power = (t - (max_pos // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.stack((scale, scale), dim = -1)
        scale = rearrange(scale, '... d r -> ... (d r)')

        return freqs, scale

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

@autocast('cuda', enabled = False)
def apply_rotary_pos_emb(t, freqs, scale = 1):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype

    freqs = freqs[-seq_len:, :]
    scale = scale[-seq_len:, :] if isinstance(scale, torch.Tensor) else scale

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = torch.cat((t, t_unrotated), dim = -1)

    return out.type(orig_dtype)

# norms

class Scale(Module):
    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        scale_fn = lambda t: t * self.value

        if not isinstance(out, tuple):
            return scale_fn(out)

        return (scale_fn(out[0]), *out[1:])

class LayerNorm(Module):
    def __init__(
        self,
        dim,
        unit_offset = False
    ):
        """
        bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
        """
        super().__init__()
        self.unit_offset = unit_offset

        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        self.gamma = nn.Parameter(torch.ones(dim))
        nn.init.constant_(self.gamma, 1. - float(unit_offset))

    def forward(self, x):
        normed = self.ln(x)
        gamma = self.gamma + float(self.unit_offset)
        return normed * gamma

class AdaptiveLayerNorm(Module):
    def __init__(
        self,
        dim,
        dim_condition = None
    ):
        super().__init__()
        dim_condition = default(dim_condition, dim)

        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        self.to_gamma = LinearNoBias(dim_condition, dim)
        nn.init.zeros_(self.to_gamma.weight)

    def forward(self, x, *, condition):
        if condition.ndim == 2:
            condition = rearrange(condition, 'b d -> b 1 d')

        normed = self.ln(x)
        gamma = self.to_gamma(condition)
        return normed * (gamma + 1.)

class ScaleNorm(Module):
    def __init__(
        self,
        dim,
        unit_offset = False
    ):
        super().__init__()
        self.unit_offset = unit_offset
        self.scale = dim ** 0.5

        self.g = nn.Parameter(torch.zeros(1))
        nn.init.constant_(self.g, 1. - float(unit_offset))

    def forward(self, x):
        gamma = self.g + float(self.unit_offset)
        return F.normalize(x, dim = -1) * self.scale * gamma

class RMSNorm(Module):
    def __init__(
        self,
        dim,
        unit_offset = False
    ):
        super().__init__()
        self.unit_offset = unit_offset
        self.scale = dim ** 0.5

        self.g = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.g, 1. - float(unit_offset))

    def forward(self, x):
        gamma = self.g + float(self.unit_offset)
        return F.normalize(x, dim = -1) * self.scale * gamma

class AdaptiveRMSNorm(Module):
    def __init__(
        self,
        dim,
        dim_condition = None
    ):
        super().__init__()
        self.scale = dim ** 0.5
        dim_condition = default(dim_condition, dim)

        self.to_gamma = LinearNoBias(dim_condition, dim)
        nn.init.zeros_(self.to_gamma.weight)

    def forward(self, x, *, condition):
        if condition.ndim == 2:
            condition = rearrange(condition, 'b d -> b 1 d')

        normed = F.normalize(x, dim = -1)
        gamma = self.to_gamma(condition)
        return normed * self.scale * (gamma + 1.)

class SimpleRMSNorm(Module):
    def __init__(
        self,
        dim,
        **kwargs
    ):
        super().__init__()
        self.scale = dim ** 0.5

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale

# residual and residual gates

class Residual(Module):
    def __init__(self, dim, scale_residual = False, scale_residual_constant = 1.):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None
        self.scale_residual_constant = scale_residual_constant

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        if self.scale_residual_constant != 1:
            residual = residual * self.scale_residual_constant

        return x + residual

class GRUGating(Module):
    def __init__(self, dim, scale_residual = False, **kwargs):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        gated_output = self.gru(
            rearrange(x, 'b n d -> (b n) d'),
            rearrange(residual, 'b n d -> (b n) d')
        )

        return gated_output.reshape_as(x)

# token shifting

def shift(t, amount, mask = None):
    if amount == 0:
        return t

    amount = min(amount, t.shape[1])

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return pad_at_dim(t, (amount, -amount), dim = - 2, value = 0.)

class ShiftTokens(Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim = -1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = [shift(*args, mask = mask) for args in zip(segments_to_shift, shifts)]
        x = torch.cat((*segments_to_shift, *rest), dim = -1)
        return self.fn(x, **kwargs)

# post branch operator

class LayerScale(Module):
    def __init__(
        self,
        fn: Module,
        dim,
        init_value = 0.,
        unit_offset = False
    ):
        super().__init__()
        self.unit_offset = unit_offset

        self.fn = fn
        self.gamma = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.gamma, init_value - float(unit_offset))

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)

        gamma = self.gamma + float(self.unit_offset)

        if isinstance(out, Tensor):
            return out * gamma

        out, *rest = out
        return out * gamma, *rest

class AdaptiveLayerScale(Module):
    def __init__(
        self,
        fn: Module,
        dim,
        dim_condition = None,
        init_bias_value = -2.
    ):
        super().__init__()
        self.fn = fn

        dim_condition = default(dim_condition, dim)
        self.to_gamma = nn.Linear(dim_condition, dim)

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.constant_(self.to_gamma.bias, init_bias_value)

    def forward(self, x, *, condition, **kwargs):
        if condition.ndim == 2:
            condition = rearrange(condition, 'b d -> b 1 d')

        out = self.fn(x, **kwargs)
        gamma = self.to_gamma(condition).sigmoid()

        if isinstance(out, Tensor):
            return out * gamma

        out, *rest = out
        return out * gamma, *rest

# skip connection combining

class ConcatCombine(Module):
    def __init__(self, dim, prev_layer_ind):
        super().__init__()
        self.prev_layer_ind = prev_layer_ind
        self.combine = LinearNoBias(dim * 2, dim)

    def forward(self, x, prev_layers: list[Tensor]):
        skip = prev_layers[self.prev_layer_ind]
        concatted_skip = torch.cat((skip, x), dim = -1)
        return self.combine(concatted_skip)

# feedforward

class GLU(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        activation: Callable,
        mult_bias = False
    ):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if mult_bias else 1.

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * self.act(gate) * self.mult_bias

class FeedForward(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        glu = False,
        glu_mult_bias = False,
        swish = False,
        relu_squared = False,
        post_act_ln = False,
        dropout = 0.,
        no_bias = False,
        zero_init_output = False
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        if glu:
            project_in = GLU(dim, inner_dim, activation, mult_bias = glu_mult_bias)
        else:
            project_in = nn.Sequential(
                nn.Linear(dim, inner_dim, bias = not no_bias),
                activation
            )

        self.ff = Sequential(
            project_in,
            LayerNorm(inner_dim) if post_act_ln else None,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias = not no_bias)
        )

        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        return self.ff(x)

# attention. it is all we need

class Attend(Module):
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        heads = None,
        pre_talking_heads = False,
        post_talking_heads = False,
        pre_scale_post_talking_heads = False,
        sparse_topk = None,
        scale = None,
        qk_norm = False,
        l2_distance = False,
        sigmoid = False,
        custom_attn_fn: Callable | None = None,
        flash = False,
        softclamp_logits = False,
        logit_softclamp_value = 50.,
        add_zero_kv = False,
        selective = False,
        hard = False,
        sigsoftmax = False,
        cope = None,
        onnxable = False,
        sdp_kwargs: dict = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True
        )
    ):
        super().__init__()
        self.scale = scale

        # causal related

        self.causal = causal
        self.create_causal_mask = onnx_create_causal_mask if onnxable else create_causal_mask

        # attention type

        assert not (flash and sigmoid), 'sigmoid attention not available for flash'
        assert not (flash and hard), 'hard attention not available for flash'
        assert at_most_one_of(sigmoid, hard, l2_distance)

        if exists(custom_attn_fn):
            self.attn_fn = custom_attn_fn
        elif sigmoid:
            self.attn_fn = F.sigmoid
        elif hard:
            self.attn_fn = one_hot_straight_through
        else:
            softmax_fn = partial(F.softmax, dim = -1)
            self.attn_fn = partial(softmax_fn, dtype = torch.float32) if not qk_norm else softmax_fn

        # dropouts

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        # talking heads

        assert not (flash and (pre_talking_heads or post_talking_heads or pre_scale_post_talking_heads)), 'talking heads not compatible with flash attention'

        self.pre_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if pre_talking_heads else None
        self.post_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if post_talking_heads else None
        self.pre_scale_post_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if pre_scale_post_talking_heads else None

        if exists(self.pre_softmax_talking_heads):
            nn.init.dirac_(self.pre_softmax_talking_heads.weight)

        if exists(self.post_softmax_talking_heads):
            nn.init.dirac_(self.post_softmax_talking_heads.weight)

        if exists(self.pre_scale_post_talking_heads):
            # an improvisation where heads are combined pre-softmax attention, then used to scale post-softmax attention
            nn.init.dirac_(self.pre_scale_post_talking_heads.weight)

        # selective attention

        assert not (flash and selective), 'selective attention cannot work on flash attention'
        assert not (selective and not causal), 'selective attention is designed for autoregressive'
        self.selective = selective

        # sparse topk

        assert not (flash and sparse_topk), 'sparse topk not compatible with flash attention'
        self.sparse_topk = sparse_topk

        # sig softmax

        assert not (flash and sigsoftmax), 'sigsoftmax not available for flash attention'
        self.sigsoftmax = sigsoftmax

        # l2 distance attention

        self.l2_distance = l2_distance

        # add a key / value token composed of zeros
        # in case this helps controlling outliers, proposed by https://www.evanmiller.org/attention-is-off-by-one.html

        self.add_zero_kv = add_zero_kv

        # soft clamp attention logit value

        if softclamp_logits:
            assert not flash, 'flash attention not compatible with logit softclamp value yet'
            assert logit_softclamp_value > 0.

        self.softclamp_logits = softclamp_logits
        self.logit_softclamp_value = logit_softclamp_value

        # contextual positional encoding

        self.cope = cope

        # flash attention

        self.flash = flash

        torch_version = version.parse(torch.__version__)
        assert not (flash and torch_version < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # torch 2.3 uses new backend and context manager

        if torch_version >= version.parse('2.3'):
            from torch.nn.attention import SDPBackend

            str_to_backend = dict(
                enable_flash = SDPBackend.FLASH_ATTENTION,
                enable_mem_efficient = SDPBackend.EFFICIENT_ATTENTION,
                enable_math = SDPBackend.MATH,
                enable_cudnn = SDPBackend.CUDNN_ATTENTION
            )

            sdpa_backends = [str_to_backend[enable_str] for enable_str, enable in sdp_kwargs.items() if enable]

            self.sdp_context_manager = partial(torch.nn.attention.sdpa_kernel, sdpa_backends)
        else:
            self.sdp_context_manager = partial(torch.backends.cuda.sdp_kernel, **sdp_kwargs)
        
        
        def custom(b, h, q_idx, kv_idx):
            return q_idx >= 0
        

    def flash_attn(
        self,
        q, k, v,
        mask = None,
        attn_bias = None
    ):
        batch, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if k.ndim == 3:
            k = repeat(k, 'b ... -> b h ...', h = q.shape[1])

        if v.ndim == 3:
            v = repeat(v, 'b ... -> b h ...', h = q.shape[1])

        # handle maybe l2 distance

        if self.l2_distance:
            k_norm_sq = k.norm(dim = -1, keepdim = True) ** 2
            k = F.pad(k, (0, 1), value = -1.)
            k = torch.cat((k, k_norm_sq), dim = -1)

            q_norm_sq = q.norm(dim = -1, keepdim = True) ** 2
            q = torch.cat((2 * q, q_norm_sq), dim = -1)
            q = F.pad(q, (0, 1), value = -1.)

        # handle scale - by default they scale by dim_head ** -0.5, but need to take care if using cosine sim attention

        if exists(self.scale):
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        causal = self.causal

        # in the case of kv caching with one token (q_len == 1), just turn off causal masking
        # in speculative decoding, this may go up to 5-6, so right aligned causal mask will be needed there

        if q_len == 1 and causal:
            causal = False

        # expand key padding mask

        if exists(mask):
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

        # handle kv cache - this should be bypassable in updated flash attention 2

        if k_len > q_len and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            if not exists(mask):
                mask = ~causal_mask
            else:
                mask = mask & ~causal_mask
            causal = False

        # manually handle causal mask, if another mask was given

        if exists(mask) and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            mask = mask & ~causal_mask
            causal = False

        # protect against an entire row being masked out

        row_is_entirely_masked = None

        if exists(mask):
            row_is_entirely_masked = ~mask.any(dim = -1)

        # handle alibi positional bias
        # convert from bool to float

        if exists(attn_bias):
            attn_bias = rearrange(attn_bias, 'h i j -> 1 h i j').expand(batch, heads, -1, -1)

            # if mask given, the mask would already contain the causal mask from above logic
            # otherwise, if no mask given but still causal, mask out alibi positional bias to a large negative number

            mask_value = -torch.finfo(q.dtype).max

            if exists(mask):
                attn_bias = attn_bias.masked_fill(~mask, mask_value // 2)
            elif causal:
                causal_mask = self.create_causal_mask(q_len, k_len, device = device)
                attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)
                causal = False

            # scaled_dot_product_attention handles attn_mask either as bool or additive bias
            # make it an additive bias here

            mask = attn_bias

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale
        with self.sdp_context_manager():
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal = False
            )

        if exists(row_is_entirely_masked) and row_is_entirely_masked.any():
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

        return out, Intermediates()

    def forward(
        self,
        q, k, v,
        mask = None,
        attn_bias = None,
        prev_attn = None
    ):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, heads, kv_heads, device = q.shape[-2], q.shape[1], k.shape[1], q.device

        scale = default(self.scale, q.shape[-1] ** -0.5)

        causal = self.causal

        # handle key padding mask

        if exists(mask) and mask.ndim == 2:
            mask = rearrange(mask, 'b j -> b 1 1 j')

        # handle kv cached decoding

        if n == 1 and causal:
            causal = False

        # handle grouped multi-query attention

        if kv_heads == 1:
            k, v = tuple(rearrange(t, 'b 1 n d -> b n d') for t in (k, v))
        elif kv_heads < heads:
            k, v = tuple(repeat(t, 'b kvh n d -> b (r kvh) n d', r = heads // kv_heads) for t in (k, v))

        # handle zero kv, as means for allowing network to attend to nothing

        if self.add_zero_kv:
            k, v = tuple(F.pad(t, (0, 0, 1, 0), value = 0.) for t in (k, v))

            if exists(mask):
                mask = F.pad(mask, (1, 0), value = True)

            if exists(attn_bias):
                attn_bias = F.pad(attn_bias, (1, 0), value = 0.)

        #if self.flash:
        #    assert not exists(prev_attn), 'residual attention not compatible with flash attention'
        return self.flash_attn(q, k, v, mask = mask, attn_bias = attn_bias)

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = DEFAULT_DIM_HEAD,
        dim_context = None,
        heads = 8,
        causal = False,
        flash = False,
        pre_talking_heads = False,
        post_talking_heads = False,
        pre_scale_post_talking_heads = False,
        head_scale = False,
        sparse_topk = None,
        num_mem_kv = 0,
        dropout = 0.,
        on_attn = False,
        gate_value_heads = False,
        swiglu_values = False,
        gate_values = False,
        zero_init_output = False,
        hard = False,
        sigsoftmax = False,
        max_attend_past = None,
        qk_norm = False,
        qk_norm_groups = 1,
        qk_norm_scale = 10,
        qk_norm_dim_scale = False,
        l2_distance = False,
        sigmoid = False,
        selective = False,
        custom_attn_fn: Callable | None = None,
        one_kv_head = False,
        kv_heads = None,
        shared_kv = False,
        value_dim_head = None,
        dim_out = None,
        tensor_product = False,      # https://arxiv.org/abs/2208.06061
        add_zero_kv = False,         # same as add_zero_attn in pytorch
        rotary_embed_values = False,
        use_cope = False,
        cope_max_pos = 16,
        cope_soft_onehot_pos = False,
        cope_talking_heads = False,
        softclamp_logits = False,
        logit_softclamp_value = 50.,
        onnxable = False
    ):
        super().__init__()
        dim_kv = default(dim_context, dim)

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.causal = causal
        self.max_attend_past = max_attend_past

        assert not (exists(kv_heads) and one_kv_head), 'either attn_one_kv_head is set to True (in which case kv_heads is set to 1), or attn_kv_heads is set, but not both'

        value_dim_head = default(value_dim_head, dim_head)
        kv_heads = default(kv_heads, heads)

        kv_heads = 1 if one_kv_head else kv_heads
        assert divisible_by(heads, kv_heads)

        self.kv_heads = kv_heads

        q_dim = dim_head * heads
        k_dim = dim_head * kv_heads
        v_dim = value_dim_head * kv_heads
        out_dim = value_dim_head * heads

        self.to_q = LinearNoBias(dim, q_dim)
        self.to_k = LinearNoBias(dim_kv, k_dim)

        # shared key / values, for further memory savings during inference

        assert not (shared_kv and value_dim_head != dim_head), 'key and value head dimensions must be equal for shared key / values'
        self.to_v = LinearNoBias(dim_kv, v_dim) if not shared_kv else None

        # relations projection from tp-attention

        self.to_r = LinearNoBias(dim, v_dim) if tensor_product else None

        # add GLU gating for aggregated values, from alphafold2

        self.to_v_gate = None
        if gate_values:
            self.to_v_gate = nn.Linear(dim, out_dim)
            self.to_v_gate_activation = F.silu if swiglu_values else F.sigmoid
            nn.init.constant_(self.to_v_gate.weight, 0)
            nn.init.constant_(self.to_v_gate.bias, 10)

        # add per head gating of the output values, from 'Attend to nothing' paper

        self.to_v_head_gate = None
        if gate_value_heads:
            self.to_v_head_gate = nn.Linear(dim, heads)
            nn.init.constant_(self.to_v_head_gate.weight, 0)
            nn.init.constant_(self.to_v_head_gate.bias, 10)

        # cosine sim attention

        self.qk_norm = qk_norm
        self.qk_norm_groups = qk_norm_groups
        self.qk_norm_scale = qk_norm_scale

        # whether to use the rmsnorm (equivalent to cosine sim attention when scale is equal to 1) - https://arxiv.org/abs/2302.05442

        self.qk_norm_dim_scale = qk_norm_dim_scale

        self.qk_norm_q_scale = self.qk_norm_k_scale = 1
        if qk_norm and qk_norm_dim_scale:
            self.qk_norm_q_scale = nn.Parameter(torch.ones(heads, 1, dim_head))
            self.qk_norm_k_scale = nn.Parameter(torch.ones(heads, 1, dim_head))

        assert (not qk_norm) or divisible_by(dim_head, qk_norm_groups), 'dimension per attention head must be divisible by the qk norm groups'
        assert not (qk_norm and (dim_head // qk_norm_groups) <= 2), 'the group dimension may be too small (2 was too small in my tests, but 4 still works, surprisingly)'

        # contextual positional encoding
        # https://arxiv.org/html/2405.18719v2

        cope = None

        if use_cope:
            assert causal, 'CoPE was designed for causal attention'
            assert not flash, 'CoPE is not flash attention compatible'

            cope = CoPE(
                dim = dim_head,
                heads = heads,
                max_pos = cope_max_pos,
                talking_heads = cope_talking_heads,
                soft_onehot = cope_soft_onehot_pos
            )

        # attend class - includes core attention algorithm + talking heads

        self.attend = Attend(
            heads = heads,
            causal = causal,
            pre_talking_heads = pre_talking_heads,
            post_talking_heads = post_talking_heads,
            pre_scale_post_talking_heads = pre_scale_post_talking_heads,
            dropout = dropout,
            sparse_topk = sparse_topk,
            hard = hard,
            qk_norm = qk_norm,
            scale = qk_norm_scale if qk_norm else self.scale,
            l2_distance = l2_distance,
            sigmoid = sigmoid,
            selective = selective,
            custom_attn_fn = custom_attn_fn,
            add_zero_kv = add_zero_kv,
            flash = flash,
            softclamp_logits = softclamp_logits,
            sigsoftmax = sigsoftmax,
            logit_softclamp_value = logit_softclamp_value,
            cope = cope,
            onnxable = onnxable
        )

        # head scaling

        self.head_scale = head_scale
        if head_scale:
            self.head_scale_params = nn.Parameter(torch.ones(1, heads, 1, 1))

        # explicit topk sparse attention

        self.sparse_topk = sparse_topk

        # add memory key / values

        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(kv_heads, num_mem_kv, dim_head))
            self.mem_v = nn.Parameter(torch.randn(kv_heads, num_mem_kv, dim_head))

        # attention on attention

        self.attn_on_attn = on_attn

        # output dimension by default same as input, but can be overridden

        dim_out = default(dim_out, dim)
        self.to_out = nn.Sequential(LinearNoBias(out_dim, dim_out * 2), nn.GLU()) if on_attn else LinearNoBias(out_dim, dim_out)

        # whether to rotate positions into values, for absolute positions in addition to relative

        self.rotary_embed_values = rotary_embed_values

        # whether parent can kv cache

        self.can_cache_kv = not selective

        # init output projection 0

        if zero_init_output:
            init_zero_(self.to_out)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        attn_mask = None,
        rel_pos = None,
        rotary_pos_emb = None,
        prev_attn = None,
        mem = None,
        mem_mask = None,
        return_intermediates = False,
        cache: Intermediates | None = None,
    ):
        b, n, h, kv_h, head_scale, num_mem_kv, device, has_context = x.shape[0], x.shape[1], self.heads, self.kv_heads, self.head_scale, self.num_mem_kv, x.device, exists(context)

        kv_input = default(context, x)

        q_input = x
        k_input = kv_input
        v_input = kv_input
        r_input = x

        if exists(mem):
            k_input, mem_packed_shape = pack([mem, k_input], 'b * d')
            v_input, _ = pack([mem, v_input], 'b * d')

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input) if exists(self.to_v) else k
        r = self.to_r(r_input) if exists(self.to_r) else None

        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        k, v, r = tuple(maybe(rearrange)(t, 'b n (h d) -> b h n d', h = kv_h) for t in (k, v, r))

        if exists(cache):
            ck, cv = cache.cached_kv

            if exists(mem):
                mk, k = unpack(k, mem_packed_shape, 'b h * d')
                mv, v = unpack(v, mem_packed_shape, 'b h * d')

            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

            if exists(mem):
                k = torch.cat((mk, k), dim = -2)
                v = torch.cat((mv, v), dim = -2)

        if return_intermediates:
            mem_len = mem.shape[-2] if exists(mem) else 0
            cached_kv = (k[..., mem_len:, :], v[..., mem_len:, :])

        if self.qk_norm:
            qk_l2norm = partial(l2norm, groups = self.qk_norm_groups)
            q, k = map(qk_l2norm, (q, k))
            scale = self.qk_norm_scale

            q = q * self.qk_norm_q_scale
            k = k * self.qk_norm_k_scale

        if exists(rotary_pos_emb) and not has_context:
            freqs, xpos_scale = rotary_pos_emb
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale ** -1.) if exists(xpos_scale) else (1., 1.)

            q = apply_rotary_pos_emb(q, freqs, q_xpos_scale)
            k = apply_rotary_pos_emb(k, freqs, k_xpos_scale)

            if self.rotary_embed_values:
                v = apply_rotary_pos_emb(v, freqs, k_xpos_scale)

        input_mask = context_mask

        if not exists(input_mask) and not has_context:
            input_mask = mask

            if (exists(input_mask) or exists(mem_mask)) and exists(mem):
                seq_len, mem_len = n, mem.shape[-2]

                if not exists(mem_mask):
                    input_mask = pad_at_dim(input_mask, (mem_len, 0), dim = -1, value = True)
                elif not exists(input_mask):
                    input_mask = pad_at_dim(mem_mask, (0, seq_len), dim = -1, value = True)
                else:
                    input_mask = torch.cat((mem_mask, input_mask), dim = -1)

        # i, j determined for relative positional bias, excluding memory key / values

        i, j = tuple(t.shape[-2] for t in (q, k))

        # maybe append memory key / values

        if num_mem_kv > 0:
            mem_k, mem_v = tuple(repeat(t, 'h n d -> b h n d', b = b) for t in (self.mem_k, self.mem_v))

            if self.qk_norm:
                mem_k = l2norm(mem_k)
                mem_k = mem_k * self.qk_norm_k_scale

            k = torch.cat((mem_k, k), dim = -2)
            v = torch.cat((mem_v, v), dim = -2)

            if exists(input_mask):
                input_mask = pad_at_dim(input_mask, (self.num_mem_kv, 0), dim = -1, value = True)

        # determine masking

        mask_value = max_neg_value(q)
        masks = []
        final_attn_mask = None

        if exists(input_mask):
            input_mask = rearrange(input_mask, 'b j -> b 1 1 j')
            masks.append(~input_mask)

        if exists(attn_mask):
            assert 2 <= attn_mask.ndim <= 4, 'attention mask must have greater than 2 dimensions but less than or equal to 4'
            if attn_mask.ndim == 2:
                attn_mask = rearrange(attn_mask, 'i j -> 1 1 i j')
            elif attn_mask.ndim == 3:
                attn_mask = rearrange(attn_mask, 'h i j -> 1 h i j')
            masks.append(~attn_mask)

        if exists(self.max_attend_past):
            range_q = torch.arange(j - i, j, device = device)
            range_k = torch.arange(j, device = device)
            dist = rearrange(range_q, 'i -> 1 1 i 1') - rearrange(range_k, 'j -> 1 1 1 j')
            max_attend_past_mask = dist > self.max_attend_past
            max_attend_past_mask = pad_at_dim(max_attend_past_mask, (num_mem_kv, 0), value = False, dim = -1) # handle memory key / values
            masks.append(max_attend_past_mask)

        if len(masks) > 0:
            final_attn_mask = ~or_reduce(masks)

        # prepare relative positional bias, if needed

        attn_bias = None
        if exists(rel_pos):
            attn_bias = rel_pos(i, j)
            attn_bias = pad_at_dim(attn_bias, (num_mem_kv, 0), value = 0.) # handle memory key / values

        # attention is all we need

        out, intermediates = self.attend(
            q, k, v,
            mask = final_attn_mask,
            attn_bias = attn_bias,
            prev_attn = prev_attn
        )

        # https://arxiv.org/abs/2208.06061 proposes to add a residual for better gradients

        if exists(r):
            out = out * r + out

        # normformer scaling of heads

        if head_scale:
            out = out * self.head_scale_params

        # per head gating, from https://arxiv.org/abs/2306.12929

        if exists(self.to_v_head_gate):
            head_gate = self.to_v_head_gate(x)
            out = out * rearrange(head_gate, 'b n h -> b h n 1').sigmoid()

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')

        # alphafold2 styled gating of the values

        if exists(self.to_v_gate):
            gates = self.to_v_gate(x)
            out = out * self.to_v_gate_activation(gates)

        # combine the heads

        out = self.to_out(out)

        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1')
            out = out.masked_fill(~mask, 0.)

        if not return_intermediates:
            return out

        intermediates.cached_kv = cached_kv

        return out, intermediates

class AttentionLayers(Module):
    def __init__(
        self,
        dim,
        depth = None,
        heads = 8,
        causal = False,
        cross_attend = False,
        only_cross = False,
        use_scalenorm = False,
        use_rmsnorm = False,
        use_simple_rmsnorm = False,
        use_adaptive_layernorm = False,
        use_adaptive_rmsnorm = False,
        use_adaptive_layerscale = False, # paired with use_adaptive_layernorm for ada-ln-zero from DiT paper
        norm_add_unit_offset = True,
        dim_condition = None,
        adaptive_condition_mlp = False,
        adaptive_condition_mlp_expansion = 4,
        alibi_pos_bias = False,
        alibi_num_heads = None,
        rel_pos_bias = False,
        rel_pos_num_buckets = 32,
        rel_pos_max_distance = 128,
        dynamic_pos_bias = False,
        dynamic_pos_bias_log_distance = False,
        dynamic_pos_bias_mlp_depth = 2,
        dynamic_pos_bias_norm = False,
        rotary_pos_emb = False,
        rotary_emb_dim = None,
        rotary_xpos = False,
        rotary_interpolation_factor = 1.,
        rotary_xpos_scale_base = 512,
        rotary_base_rescale_factor = 1.,
        weight_tie_layers = False,
        custom_layers: tuple[str, ...] | None = None,
        layers_execute_order: tuple[int, ...] | None = None,
        sandwich_coef = None,
        par_ratio = None,
        residual_attn = False,
        cross_residual_attn = False,
        macaron = False,
        pre_norm = True,
        pre_norm_has_final_norm = True,
        gate_residual = False,
        scale_residual = False,
        scale_residual_constant = 1.,
        shift_tokens = 0,
        sandwich_norm = False,
        softclamp_output = False,
        softclamp_output_value = 30.,
        resi_dual = False,
        resi_dual_scale = 1.,
        zero_init_branch_output = False,
        layer_dropout = 0.,
        cross_attn_tokens_dropout = 0.,
        disable_abs_pos_emb = None,
        use_layerscale = False,
        layerscale_init_value = 0.,
        unet_skips = False,
        **kwargs
    ):
        super().__init__()
        rotary_pos_emb = rotary_pos_emb or rotary_xpos

        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, kwargs = groupby_prefix_and_trim('attn_', kwargs)
        cross_attn_kwargs, kwargs = groupby_prefix_and_trim('cross_attn_', kwargs)

        assert len(kwargs) == 0, f'unrecognized kwargs passed in {kwargs.keys()}'

        dim_head = attn_kwargs.get('dim_head', DEFAULT_DIM_HEAD)

        self.dim = dim
        self.causal = causal
        self.layers = ModuleList([])

        self.disable_abs_pos_emb = default(disable_abs_pos_emb, (rel_pos_bias or rotary_pos_emb))

        rotary_emb_dim = max(default(rotary_emb_dim, dim_head // 2), 32)

        assert not (rotary_xpos and not causal), 'rotary xpos is not compatible with bidirectional attention'
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim, use_xpos = rotary_xpos, scale_base = rotary_xpos_scale_base, interpolation_factor = rotary_interpolation_factor, base_rescale_factor = rotary_base_rescale_factor) if rotary_pos_emb else None

        assert not (alibi_pos_bias and rel_pos_bias), 'you can only choose Alibi positional bias or T5 relative positional bias, not both'
        assert rel_pos_num_buckets <= rel_pos_max_distance, 'number of relative position buckets must be less than the relative position max distance'

        # relative positional bias

        flash_attn = attn_kwargs.get('flash', False)
        assert at_most_one_of(rel_pos_bias, dynamic_pos_bias, alibi_pos_bias), 'you can only choose up to one of t5, alibi, or dynamic positional bias'

        self.rel_pos = None

        if rel_pos_bias:
            assert not flash_attn, 'flash attention not compatible with t5 relative positional bias'
            self.rel_pos = RelativePositionBias(scale = dim_head ** 0.5, causal = causal, heads = heads, num_buckets = rel_pos_num_buckets, max_distance = rel_pos_max_distance)
        elif dynamic_pos_bias:
            assert not flash_attn, 'flash attention not compatible with dynamic positional bias'
            self.rel_pos = DynamicPositionBias(dim = dim // 4, heads = heads, log_distance = dynamic_pos_bias_log_distance, depth = dynamic_pos_bias_mlp_depth, norm = dynamic_pos_bias_norm)
        elif alibi_pos_bias:
            alibi_num_heads = default(alibi_num_heads, heads)
            assert alibi_num_heads <= heads, 'number of ALiBi heads must be less than the total number of heads'
            self.rel_pos = AlibiPositionalBias(heads = alibi_num_heads, total_heads = heads)

        assert at_most_one_of(sandwich_norm, resi_dual), 'either sandwich norm or resiDual is selected, but not both'
        assert not (not pre_norm and sandwich_norm), 'sandwich norm cannot be used when not using prenorm'

        if resi_dual:
            pre_norm = False

        self.pre_norm = pre_norm
        self.sandwich_norm = sandwich_norm

        self.resi_dual = resi_dual
        assert 0 < resi_dual_scale <= 1., 'resiDual prenorm residual must be scaled by a factor greater than 0 and less than or equal to 1.'
        self.resi_dual_scale = resi_dual_scale

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn
        assert not (flash_attn and (residual_attn or cross_residual_attn)), 'flash attention is not compatible with residual attention'

        self.cross_attend = cross_attend

        # determine norm

        assert at_most_one_of(use_scalenorm, use_rmsnorm, use_simple_rmsnorm, use_adaptive_layernorm, use_adaptive_rmsnorm), 'you can only use either scalenorm, rmsnorm, adaptive layernorm, adaptive rmsnorm, or simple rmsnorm'

        norm_need_condition = False
        dim_condition = default(dim_condition, dim)
        dim_condition_mult = 1

        if adaptive_condition_mlp:
            dim_condition_mult = adaptive_condition_mlp_expansion

        if use_scalenorm:
            norm_class = ScaleNorm
        elif use_rmsnorm:
            norm_class = RMSNorm
        elif use_simple_rmsnorm:
            norm_class = SimpleRMSNorm
        elif use_adaptive_layernorm:
            norm_need_condition = True
            norm_class = partial(AdaptiveLayerNorm, dim_condition = dim_condition * dim_condition_mult)
        elif use_adaptive_rmsnorm:
            norm_need_condition = True
            norm_class = partial(AdaptiveRMSNorm, dim_condition = dim_condition * dim_condition_mult)
        else:
            norm_class = LayerNorm

        norm_fn = partial(norm_class, dim)

        if not norm_need_condition and norm_add_unit_offset:
            # researcher Ohad Rubin shares in a blog post by adding an offset to gammas, they can be subjected to weight decay safely
            norm_fn = partial(norm_fn, unit_offset = True)

        self.norm_need_condition = norm_need_condition
        self.dim_condition = dim_condition

        # determine default block layer type order

        if cross_attend and not only_cross:
            default_block = ('a', 'c', 'f')
        elif cross_attend and only_cross:
            default_block = ('c', 'f')
        else:
            default_block = ('a', 'f')

        if macaron:
            default_block = ('f',) + default_block

        # determine post branch wrapper

        assert at_most_one_of(use_layerscale, use_adaptive_layerscale)

        post_branch_fn = None
        post_branch_fn_needs_condition = False

        if use_layerscale:
            post_branch_fn = partial(LayerScale, dim = dim, init_value = layerscale_init_value)
        elif use_adaptive_layerscale:
            post_branch_fn = partial(AdaptiveLayerScale, dim = dim, dim_condition = dim_condition * dim_condition_mult)
            post_branch_fn_needs_condition = True

        self.post_branch_fn_needs_condition = post_branch_fn_needs_condition

        if exists(post_branch_fn) and not post_branch_fn_needs_condition and norm_add_unit_offset:
            post_branch_fn = partial(post_branch_fn, unit_offset = True)

        # setup mlp for conditioning

        self.need_condition = norm_need_condition or post_branch_fn_needs_condition

        self.adaptive_mlp = nn.Identity()

        if self.need_condition and adaptive_condition_mlp:
            self.adaptive_mlp = nn.Sequential(
                LinearNoBias(dim_condition, dim_condition * dim_condition_mult),
                nn.SiLU()
            )

        # zero init

        if zero_init_branch_output:
            attn_kwargs = {**attn_kwargs, 'zero_init_output':  True}
            ff_kwargs = {**ff_kwargs, 'zero_init_output':  True}

        # setup weight tying, which is a special case of `layer_execute_order`

        assert not (exists(layers_execute_order) and exists(custom_layers) and exists(depth)), 'depth should not be passed in if using custom layers and custom layer execution order'

        assert not (weight_tie_layers and any([*map(exists, (custom_layers, par_ratio, sandwich_coef))]))

        if weight_tie_layers:
            assert exists(depth), 'depth must be passed in with `weight_tie_layers` = True'
            assert not exists(layers_execute_order)
            layers_execute_order = tuple(range(len(default_block))) * depth
            depth = 1

        # calculate layer block order

        len_default_block = 1

        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, 'par ratio out of range'
            default_block = tuple(filter(not_equals('f'), default_block))
            par_attn  = par_depth // par_ratio
            depth_cut = par_depth * 2 // 3  # 2 / 3 attention layer cutoff suggested by PAR paper
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert len(default_block) <= par_width, 'default block is too large for par_ratio'
            par_block = default_block + ('f',) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ('f',) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert sandwich_coef > 0 and sandwich_coef <= depth, 'sandwich coefficient should be less than the depth'
            layer_types = ('a',) * sandwich_coef + default_block * (depth - sandwich_coef) + ('f',) * sandwich_coef
        else:
            assert exists(depth), '`depth` must be passed in for `Decoder` or `Encoder`'
            layer_types = default_block * depth
            len_default_block = len(default_block)

        self.layer_types = layer_types
        self.layers_execute_order = default(layers_execute_order, tuple(range(len(layer_types))))

        assert all([i < len(self.layer_types) for i in self.layers_execute_order])

        self.num_attn_layers = len(list(filter(equals('a'), layer_types)))

        # set the depth

        depth = default(depth, len(self.layers_execute_order))
        self.depth = depth

        # stochastic depth

        self.layer_dropouts = cast_tuple(layer_dropout, len(layer_types))

        # structured dropout for cross attending

        self.cross_attn_tokens_dropout = cross_attn_tokens_dropout

        # calculate token shifting

        shift_tokens = cast_tuple(shift_tokens, len(layer_types))

        # optional soft clamping just before the final norm
        # used in gemma 2

        self.softclamp_output = softclamp_output
        self.softclamp_output_value = softclamp_output_value

        # whether it has post norm

        self.final_norm = norm_fn() if pre_norm or resi_dual else nn.Identity()

        # whether unet or not

        self.unet_skips = unet_skips
        num_skips = self.depth // len_default_block

        assert not (unet_skips and num_skips == 0), 'must have depth of at least 2 for unet skip connections'

        skip_indices = [i * len_default_block for i in range(num_skips)]

        self.skip_combines = ModuleList([])

        # iterate and construct layers

        for ind, (layer_type, layer_shift_tokens) in enumerate(zip(self.layer_types, shift_tokens)):

            # `ind` is the index of each module - attention, feedforward, cross attention
            # but `block_ind` refers to the typical enumeration of a transformer block (attn + ff + [optional] cross attn)

            block_begin = divisible_by(ind, len_default_block)
            block_ind = ind // len_default_block

            is_last_layer = ind == (len(self.layer_types) - 1)

            # attention, cross attention, feedforward

            if layer_type == 'a':
                layer = Attention(dim, heads = heads, causal = causal, **attn_kwargs)
            elif layer_type == 'c':
                layer = Attention(dim, heads = heads, **{**attn_kwargs, **cross_attn_kwargs})
            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f'invalid layer type {layer_type}')

            if layer_shift_tokens > 0:
                shift_range_upper = layer_shift_tokens + 1
                shift_range_lower = -layer_shift_tokens if not causal else 0
                layer = ShiftTokens(range(shift_range_lower, shift_range_upper), layer)

            if exists(post_branch_fn):
                layer = post_branch_fn(layer)

            residual_fn = GRUGating if gate_residual else Residual

            residual = residual_fn(dim, scale_residual = scale_residual, scale_residual_constant = scale_residual_constant)

            # handle unet skip connection

            skip_combine = None
            is_latter_half = block_begin and block_ind >= (self.depth / 2)

            if self.unet_skips and is_latter_half:
                skip_combine = ConcatCombine(dim, skip_indices.pop())

            # all normalizations of the layer

            pre_branch_norm = norm_fn() if pre_norm else None
            post_branch_norm = norm_fn() if sandwich_norm else None
            post_main_norm = norm_fn() if not pre_norm else None

            norms = ModuleList([
                pre_branch_norm,
                post_branch_norm,
                post_main_norm
            ])

            self.skip_combines.append(skip_combine)

            self.layers.append(ModuleList([
                norms,
                layer,
                residual
            ]))

        # determine whether can cache kv

        self.can_cache_kv = all([module.can_cache_kv for module in self.modules() if isinstance(module, Attention) ])

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        attn_mask = None,
        self_attn_kv_mask = None,
        mems = None,
        mem_masks = None,
        seq_start_pos: Tensor | None = None,
        cache: LayerIntermediates | None = None,
        cache_age = 1,
        return_hiddens = False,
        rotary_pos_emb = None,
        condition = None,
        layers_execute_order: tuple[int, ...] | None = None
    ):
        assert not (self.cross_attend ^ exists(context)), 'context must be passed in if cross_attend is set to True'
        assert not (exists(condition) ^ self.need_condition), 'condition needs to be passed in if using adaptive layernorm or vice versa'

        # handle condition

        if exists(condition):
            assert condition.shape[-1] == self.dim_condition, f'expected condition dimension of {self.dim_condition} but received {condition.shape[-1]}'

            assert condition.ndim in {2, 3}

            if condition.ndim == 2:
                condition = rearrange(condition, 'b d -> b 1 d')

            condition = self.adaptive_mlp(condition)

        # setup maybe layernorm kwarg

        norm_kwargs = dict()

        if self.norm_need_condition:
            norm_kwargs.update(condition = condition)

        # maybe post branch fn conditioning (DiT paper's ada-ln-zero)

        block_forward_kwargs = dict()

        if self.post_branch_fn_needs_condition:
            block_forward_kwargs.update(condition = condition)

        # initialize accums

        hiddens = []
        layer_hiddens = []
        intermediates = []

        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers
        mem_masks = mem_masks.copy() if exists(mem_masks) else [None] * self.num_attn_layers

        # handle left padded sequences

        if exists(seq_start_pos):
            seq_arange = torch.arange(x.shape[-2], device = x.device, dtype = torch.long)
            left_pad_mask = seq_arange >= seq_start_pos[..., None]

            if exists(self_attn_kv_mask):
                self_attn_kv_mask = self_attn_kv_mask & left_pad_mask
            else:
                self_attn_kv_mask = left_pad_mask

        # rotary positions

        if not exists(rotary_pos_emb) and exists(self.rotary_pos_emb):
            maybe_mem = mems[0] # todo - handle edge case where different layers get different memory lengths. don't think this will ever come up but who knows
            mem_len = maybe_mem.shape[1] if exists(maybe_mem) else 0

            pos = torch.arange(x.shape[1] + mem_len, device = x.device) - mem_len
            rotary_pos_emb = self.rotary_pos_emb(pos)

        # assume cached key / values

        attn_cache = []

        if exists(cache):
            #assert not self.training and self.causal and not any([*map(exists, (mask, attn_mask))])

            if exists(context):
                context = context[:, :0]

            if cache_age > 0:
                x = x[:, -cache_age:] # for spec decoding, may be greater than 1

            attn_cache = cache.attn_intermediates

        iter_attn_cache = iter(attn_cache)

        # outer residual - for resiDual paper

        outer_residual = x * self.resi_dual_scale

        # get layers to be executed

        layer_variables = (
            self.layer_types,
            self.skip_combines,
            self.layers,
            self.layer_dropouts
        )

        # able to override the layers execution order on forward, for trying to depth extrapolate

        layers_execute_order = default(layers_execute_order, self.layers_execute_order)

        layer_variables = tuple(tuple(layer_variable[i] for i in layers_execute_order) for layer_variable in layer_variables)

        # store all hiddens for skips

        skip_hiddens = []

        # go through the attention and feedforward layers

        for ind, (layer_type, skip_combine, (norm, block, residual_fn), layer_dropout) in enumerate(zip(*layer_variables)):
            is_last = ind == (len(self.layers) - 1)

            # handle skip connections

            skip_hiddens.append(x)

            if exists(skip_combine):
                x = skip_combine(x, skip_hiddens)

            # layer dropout

            if self.training and layer_dropout > 0. and random() < layer_dropout:
                continue

            if layer_type == 'a':
                if return_hiddens:
                    hiddens.append(x)

                layer_mem = mems.pop(0) if mems else None
                layer_mem_mask = mem_masks.pop(0) if mem_masks else None

            if layer_type == 'c':
                if self.training and self.cross_attn_tokens_dropout > 0.:
                    context, context_mask = dropout_seq(context, context_mask, self.cross_attn_tokens_dropout)

            inner_residual = x

            if return_hiddens:
                layer_hiddens.append(x)

            pre_norm, post_branch_norm, post_main_norm = norm

            if self.need_condition:
                pre_norm = maybe(partial)(pre_norm, **norm_kwargs)
                post_branch_norm = maybe(partial)(post_branch_norm, **norm_kwargs)
                post_main_norm = maybe(partial)(post_main_norm, **norm_kwargs)

            if exists(pre_norm):
                x = pre_norm(x)

                if layer_type == 'a' and exists(layer_mem):
                    layer_mem = pre_norm(layer_mem)

            block = partial(block, **block_forward_kwargs)

            if layer_type == 'a':
                out, inter = block(x, mask = mask, context_mask = self_attn_kv_mask, attn_mask = attn_mask, rel_pos = self.rel_pos, rotary_pos_emb = rotary_pos_emb, prev_attn = prev_attn, cache = next(iter_attn_cache, None), mem = layer_mem, mem_mask = layer_mem_mask, return_intermediates = True)
            elif layer_type == 'c':
                out, inter = block(x, context = context, mask = mask, context_mask = context_mask, prev_attn = prev_cross_attn, cache = next(iter_attn_cache, None), return_intermediates = True)
            elif layer_type == 'f':
                out = block(x)

            if self.resi_dual:
                outer_residual = outer_residual + out * self.resi_dual_scale

            if exists(post_branch_norm):
                out = post_branch_norm(out)

            x = residual_fn(out, inner_residual)

            if layer_type in ('a', 'c') and return_hiddens:
                inter.layer_type = layer_type
                intermediates.append(inter)

            if layer_type == 'a' and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == 'c' and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if exists(post_main_norm):
                x = post_main_norm(x)

        if return_hiddens:
            layer_hiddens.append(x)

        if self.softclamp_output:
            x = softclamp(x, self.softclamp_output_value)

        final_norm = self.final_norm

        if self.need_condition:
            final_norm = maybe(partial)(final_norm, **norm_kwargs)

        if self.resi_dual:
            x = x + final_norm(outer_residual)
        else:
            x = final_norm(x)

        if not return_hiddens:
            return x

        intermediates = LayerIntermediates(
            hiddens = hiddens,
            last_hidden = x,
            attn_intermediates = intermediates,
            layer_hiddens = layer_hiddens,
        )

        return x, intermediates

class FlashAttentionTransformers(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on encoder'
        super().__init__(causal = False, **kwargs)

