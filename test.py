import torch
from attention import MultiheadAttention
from torch import nn
import logging

torch.manual_seed(1)
device = 'cuda:1'
# logging.basicConfig(filename='me_attn_output.log', format='%(message)s', level=logging.INFO)

l, b, embed_dim = 1024, 32, 128
tgt_len = 1024
q = torch.randn((l, b, embed_dim), requires_grad=True, device=device)
k = torch.randn((tgt_len, b, embed_dim), requires_grad=True, device=device)
v = torch.randn((tgt_len, b, embed_dim), requires_grad=True, device=device)

# origin_msa = MultiheadAttention(embed_dim, 4).to(device)
me_msa = MultiheadAttention(embed_dim, 4, chunksize_q=1, chunksize_k=512).to(device)
# torch_msa = nn.MultiheadAttention(embed_dim, 4).to(device)

attn_output = me_msa(q, k, v)
print(torch.cuda.max_memory_allocated(device=device))
attn_output.backward(torch.ones_like(attn_output))
print(torch.cuda.max_memory_allocated(device=device))
# logging.info(attn_output)
# logging.info(q.grad)
# logging.info(k.grad)
# logging.info(v.grad)
