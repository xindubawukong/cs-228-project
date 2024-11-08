#ifndef LINEAR_H_
#define LINEAR_H_

#include <string>

#include "utils.h"

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
  ~Linear() {
    if (weight) delete[] weight;
    if (bias) delete[] bias;
  }

  void LoadParameters(auto& params, std::string prefix) {
    Load(weight, params[prefix + ".weight"], in * out);
    Load(bias, params[prefix + ".bias"], out);
  }

  void Forward(float* x, int m, float* y) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < out; j++) {
        y[i * out + j] = bias[j];
        for (int k = 0; k < in; k++) {
          y[i * out + j] += x[i * in + k] * y[k * out + j];
        }
      }
    }
  }
};

#endif  // LINEAR_H_
