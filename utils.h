#ifndef UTILS_H_
#define UTILS_H_

#include "torch/script.h"

template <typename T>
void Load(T*& dst, at::Tensor& tensor, size_t n) {
  size_t tot = 1;
  for (int i = 0; i < tensor.dim(); i++) {
    tot *= tensor.size(i);
  }
  assert(tot == n);
  if (dst == nullptr) {
    dst = new T[n];
  }
  T* src = (T*)tensor.data_ptr();
  std::copy(src, src + n, dst);
}

#endif  // UTILS_H_
