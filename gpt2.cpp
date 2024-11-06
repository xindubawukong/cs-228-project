#include <torch/script.h>

#include <iostream>
#include <nlohmann/json.hpp>

#include "gpt2_config.h"
#include "utils.h"

using namespace std;

using json = nlohmann::json;

/*
def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b
*/
struct Linear {
  int in, out;
  float* weight;
  float* bias;
  Linear(int in_, int out_)
      : in(in_), out(out_), weight(nullptr), bias(nullptr) {}
  void LoadParameters(auto& params, std::string prefix) {
    Load(weight, params[prefix + ".weight"], in * out);
    Load(bias, params[prefix + ".bias"], out);
  }
};

/*
def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + eps)  # normalize x to have mean=0 and var=1 over last axis
    return g * x + b  # scale and offset with gamma/beta params
*/
struct LayerNorm {
  int n_embd;
  float* weight;
  float* bias;
  LayerNorm(int n_embd_) : n_embd(n_embd_), weight(nullptr), bias(nullptr) {}
  void LoadParameters(auto& params, string prefix) {
    Load(weight, params[prefix + ".weight"], n_embd);
    Load(bias, params[prefix + ".bias"], n_embd);
  }
};

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
struct Attention {
  int n_embd;
  Linear c_attn;
  Linear c_proj;
  Attention(int n_embd_)
      : n_embd(n_embd_), c_attn(n_embd, 3 * n_embd), c_proj(n_embd, n_embd) {}
  void LoadParameters(auto& params, string prefix) {
    c_attn.LoadParameters(params, prefix + ".c_attn");
    c_proj.LoadParameters(params, prefix + ".c_proj");
  }
};

/*
def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # project up
    a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]

    # project back down
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

    return x
*/
struct MLP {
  int n_embd;
  Linear c_fc;
  Linear c_proj;
  MLP(int n_embd_)
      : n_embd(n_embd_), c_fc(n_embd, 4 * n_embd), c_proj(4 * n_embd, n_embd) {}
  void LoadParameters(auto& params, string prefix) {
    c_fc.LoadParameters(params, prefix + ".c_fc");
    c_proj.LoadParameters(params, prefix + ".c_proj");
  }
};

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
  Attention attn;
  LayerNorm ln_2;
  MLP mlp;

  TransformerBlock(GPT2Config config_)
      : config(config_),
        layer_id(-1),
        ln_1(config.n_embd),
        attn(config.n_embd),
        ln_2(config.n_embd),
        mlp(config.n_embd) {}

  void LoadParameters(auto& params, string prefix) {
    ln_1.LoadParameters(params, prefix + ".ln_1");
    attn.LoadParameters(params, prefix + ".attn");
    ln_2.LoadParameters(params, prefix + ".ln_2");
    mlp.LoadParameters(params, prefix + ".mlp");
  }
};

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
    std::cout << "Loading GPT2 parameters" << std::endl;
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
    std::cout << "Successfully loaded GPT2 parameters" << std::endl;
  }
};

int main() {
  torch::jit::script::Module gpt2_torch =
      torch::jit::load("../model/gpt2_model.pt");

  GPT2Config config = json::parse(ifstream("../model/gpt2_config.json"));

  GPT2 gpt2(config);

  unordered_map<string, at::Tensor> params;
  for (const auto& param : gpt2_torch.named_parameters()) {
    std::cout << param.name << ' ' << param.value.sizes() << std::endl;
    params[param.name] = param.value;
  }
  gpt2.LoadParameters(params);

  // Example input
  torch::Tensor input_ids = torch::randint(0, 50257, {1, 20}, torch::kLong);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input_ids);

  // (logits, past_key_values, hidden_states)
  auto output = gpt2_torch.forward(inputs).toTuple();

  auto t = output->elements()[0].toTensor();
  cout << t.sizes() << endl;
  auto x = output->elements()[1].toTuple();
  cout << x->size() << endl;

  return 0;
}
