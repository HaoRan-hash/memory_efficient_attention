## Memory_Efficient_Attention

This is  an unofficial implementation of [Self-attention Does Not Need O(n^2) Memory](https://arxiv.org/abs/2112.05682v2) for Pytorch. The API interface is designed in the same style as `torch.nn.MultiheadAttention`, but not support `dropout` and `mask`. Example of use is in `test.py`.

The algorithm is good when `q,k,v` length is very long (e.g. more than 2^12). In practice, I suggest you use small `chunksize_q` and relatively large `chunksize_k`, this is the best setting to save GPU memory. If you want to learn more about the algorithm, please read the original article.

### Citation

```markdown
@misc{rabe2021selfattention,
      title={Self-attention Does Not Need $O(n^2)$ Memory}, 
      author={Markus N. Rabe and Charles Staats},
      year={2021},
      eprint={2112.05682},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
