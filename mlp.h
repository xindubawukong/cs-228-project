#ifndef MLP_H_
#define MLP_H_

#include <string>

#include "linear.h"
#include "utils.h"

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

  void LoadParameters(auto& params, std::string prefix) {
    c_fc.LoadParameters(params, prefix + ".c_fc");
    c_proj.LoadParameters(params, prefix + ".c_proj");
  }
};

#endif  // MLP_H_
