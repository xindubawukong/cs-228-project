#ifndef TRANSFORMER_BLOCK_H_
#define TRANSFORMER_BLOCK_H_

#include <string>

#include "gpt2_config.h"
#include "layer_norm.h"
#include "mlp.h"
#include "multi_head_attention.h"
#include "utils.h"

/*
def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # multi-head causal self attention
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x
*/
struct TransformerBlock {
  GPT2Config config;
  int layer_id;
  LayerNorm ln_1;
  MultiHeadAttention attn;
  LayerNorm ln_2;
  MLP mlp;

  TransformerBlock(GPT2Config config_)
      : config(config_),
        layer_id(-1),
        ln_1(config.n_embd),
        attn(config.n_embd, config.n_head),
        ln_2(config.n_embd),
        mlp(config.n_embd) {}

  void LoadParameters(auto& params, std::string prefix) {
    ln_1.LoadParameters(params, prefix + ".ln_1");
    attn.LoadParameters(params, prefix + ".attn");
    ln_2.LoadParameters(params, prefix + ".ln_2");
    mlp.LoadParameters(params, prefix + ".mlp");
  }
};

#endif  // TRANSFORMER_BLOCK_H_
