#ifndef MULTI_HEAD_ATTENTION_H_
#define MULTI_HEAD_ATTENTION_H_

#include <string>

#include "linear.h"
#include "utils.h"

/*
def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projection
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    qkv = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

    # split into heads
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]

    # causal mask to hide future inputs from being attended to
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]

    # perform attention over each head
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]  # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]

    # merge heads
    x = np.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x
*/
struct MultiHeadAttention {
  int n_embd;
  int n_head;
  Linear c_attn;
  Linear c_proj;

  MultiHeadAttention(int n_embd_, int n_head_)
      : n_embd(n_embd_),
        n_head(n_head_),
        c_attn(n_embd, 3 * n_embd),
        c_proj(n_embd, n_embd) {}

  void LoadParameters(auto& params, std::string prefix) {
    c_attn.LoadParameters(params, prefix + ".c_attn");
    c_proj.LoadParameters(params, prefix + ".c_proj");
  }
};

#endif  // MULTI_HEAD_ATTENTION_H_
