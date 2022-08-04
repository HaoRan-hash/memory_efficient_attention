import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.init import xavier_uniform_, constant_
import math


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, chunksize_q=None, chunksize_k=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = (self.embed_dim == self.kdim == self.vdim)

        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'

        if chunksize_q and chunksize_k:
            self.memory_efficient = True
            self.chunksize_q = chunksize_q
            self.chunksize_k = chunksize_k
        else:
            self.memory_efficient = False
        
        if self._qkv_same_embed_dim:
            self.in_proj_weight = nn.parameter.Parameter(torch.empty((3 * embed_dim, embed_dim), dtype=torch.float32))
        else :
            self.q_proj_weight = nn.parameter.Parameter(torch.empty((embed_dim, embed_dim), dtype=torch.float32))
            self.k_proj_weight = nn.parameter.Parameter(torch.empty((embed_dim, self.kdim), dtype=torch.float32))
            self.v_proj_weight = nn.parameter.Parameter(torch.empty((embed_dim, self.vdim), dtype=torch.float32))
        
        self.in_proj_bias = nn.parameter.Parameter(torch.empty((3 * embed_dim, ), dtype=torch.float32))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()
    
    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)
        
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)
    
    def _in_projection_packed(self, q, k, v):
        if k is v:
            if q is k:
                return F.linear(q, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
            else:
                w_q, w_kv = self.in_proj_weight.split([self.embed_dim, self.embed_dim * 2])
                b_q, b_kv = self.in_proj_bias.split([self.embed_dim, self.embed_dim * 2])
                return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
        else:
            w_q, w_k, w_v = self.in_proj_weight.chunk(3)
            b_q, b_k, b_v = self.in_proj_bias.chunk(3)
            return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

    def _in_projection(self, q, k, v):
        b_q, b_k, b_v = self.in_proj_bias.chunk(3)
        return F.linear(q, self.q_proj_weight, b_q), F.linear(k, self.k_proj_weight, b_k), F.linear(v, self.v_proj_weight, b_v)
    
    def _attention(self, q, k, v):
        q = q / math.sqrt(self.head_dim)
        attn_map = torch.bmm(q, k.transpose(-2, -1))
        attn_map = attn_map.softmax(dim=-1)
        return torch.bmm(attn_map, v)
    
    def _chunk_compute(self, chunk_q, chunk_k, chunk_v):
        chunk_attn_map = torch.bmm(chunk_q, chunk_k.transpose(-2, -1))

        # 为了exp不溢出做的numerical stability处理
        chunk_attn_map_max, _ = chunk_attn_map.max(dim=-1, keepdim=True)
        chunk_attn_map_max = chunk_attn_map_max.detach()
        exp_chunk_attn_map = torch.exp(chunk_attn_map-chunk_attn_map_max)

        row_sum = exp_chunk_attn_map.sum(dim=-1)
        return torch.bmm(exp_chunk_attn_map, chunk_v), row_sum, chunk_attn_map_max.squeeze(-1)

    def _memory_efficient_attention(self, q, k, v):
        src_len = q.shape[1]
        tgt_len = k.shape[1]
        q = q / math.sqrt(self.head_dim)

        assert src_len % self.chunksize_q == 0, 'q length must be divisible by chunksize_q'
        assert tgt_len % self.chunksize_k == 0, 'k length must be divisible by chunksize_k'
        attn_outputs = []

        for i in range(src_len // self.chunksize_q):
            chunk_outputs = []
            chunk_q = q[:, i * self.chunksize_q:(i + 1) * self.chunksize_q, :]
            for j in range(tgt_len // self.chunksize_k):
                chunk_k = k[:, j * self.chunksize_k:(j + 1) * self.chunksize_k, :]
                chunk_v = v[:, j * self.chunksize_k:(j + 1) * self.chunksize_k, :]
                if self.training:
                    chunk_outputs.append(checkpoint(self._chunk_compute, chunk_q, chunk_k, chunk_v))
                else:
                    chunk_outputs.append(self._chunk_compute(chunk_q, chunk_k, chunk_v))
            
            chunk_values, chunk_weights, chunk_max = list(map(torch.stack, zip(*chunk_outputs)))
            global_max, _ = chunk_max.max(dim=0, keepdim=True)
            max_diffs = torch.exp(chunk_max - global_max)
            chunk_values *= max_diffs.unsqueeze(-1)
            chunk_weights *= max_diffs

            all_values = chunk_values.sum(0)
            all_weights = chunk_weights.unsqueeze(dim=-1).sum(0)
            attn_outputs.append(all_values / all_weights)
        
        attn_outputs = torch.cat(attn_outputs, dim=1)
        return attn_outputs
    
    def forward(self, q, k, v):
        """
        q.shape = (l, b, embed_dim)
        k.shape = (l', b, kdim)
        v.shape = (l', b, vdim)
        """
        _, b, _ = q.shape
        if self._qkv_same_embed_dim:
            q, k, v = self._in_projection_packed(q, k, v)
        else:
            q, k, v = self._in_projection(q, k, v)
        q = q.unflatten(2, (self.num_heads, self.head_dim)).flatten(1, 2).transpose(0, 1)   # q.shape = (b * num_heads, l, head_dim)
        k = k.unflatten(2, (self.num_heads, self.head_dim)).flatten(1, 2).transpose(0, 1)
        v = v.unflatten(2, (self.num_heads, self.head_dim)).flatten(1, 2).transpose(0, 1)

        if self.memory_efficient:
            attn_output = self._memory_efficient_attention(q, k, v)   # attn_output.shape = (b * num_heads, l, head_dim)
        else:
            attn_output = self._attention(q, k, v)
        
        attn_output = attn_output.transpose(0, 1).unflatten(1, (b, self.num_heads)).flatten(-2, -1)   # attn_output.shape = (l, b, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output
