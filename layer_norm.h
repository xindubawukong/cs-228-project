#ifndef LAYER_NORM_H_
#define LAYER_NORM_H_

#include <string>

#include "utils.h"

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

  void LoadParameters(auto& params, std::string prefix) {
    Load(weight, params[prefix + ".weight"], n_embd);
    Load(bias, params[prefix + ".bias"], n_embd);
  }
};

#endif  // LAYER_NORM_H_
