#ifndef GPT2_H_
#define GPT2_H_

#include "gpt2_config.h"
#include "transformer_block.h"
#include "utils.h"

/*
def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):  # [n_seq] -> [n_seq, n_vocab]
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]

    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]
*/
struct GPT2 {
  GPT2Config config;
  float* wte;
  float* wpe;
  std::vector<TransformerBlock> blocks;
  LayerNorm ln_f;
  float* lm_head;

  GPT2(GPT2Config config_)
      : config(config_),
        wte(nullptr),
        wpe(nullptr),
        ln_f(config.n_embd),
        lm_head(nullptr) {
    blocks.assign(config.n_layer, TransformerBlock(config));
    for (int i = 0; i < config.n_layer; i++) {
      blocks[i].layer_id = i;
    }
  }

  void LoadParameters(auto& params) {
    std::cout << "\nLoading GPT2 parameters" << std::endl;
    Load(wte, params["model.transformer.wte.weight"],
         config.vocab_size * config.n_embd);
    Load(wpe, params["model.transformer.wpe.weight"],
         config.n_ctx * config.n_embd);
    for (auto& block : blocks) {
      block.LoadParameters(
          params, "model.transformer.h." + std::to_string(block.layer_id));
    }
    ln_f.LoadParameters(params, "model.transformer.ln_f");
    Load(lm_head, params["model.lm_head.weight"],
         config.vocab_size * config.n_embd);
    std::cout << "Successfully loaded GPT2 parameters\n" << std::endl;
  }
};

#endif  // GPT2_H_
