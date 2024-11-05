
## Requirements

- libtorch
  - Download from https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.5.1%2Bcpu.zip
  - Put libtorch at `third_party/libtorch`


The error you're encountering is caused by `torch.jit.script` trying to script a Hugging Face model that internally uses certain Python-specific constructs (like `_lru_cache_wrapper`), which are not compatible with TorchScript. The Hugging Face models, such as `GPT2LMHeadModel`, often include features that TorchScript cannot script directly due to dynamic attributes or caching mechanisms that TorchScript does not support.

To work around this issue, you can use `torch.jit.trace` on just the parts of the model that don’t require `None` handling, by splitting the logic for handling `past_key_values`. Here's a refined approach:

### Workaround: Split the Model Logic for `None` Handling

Since `torch.jit.trace` does not handle `None` well, we can:
1. Implement two functions within the wrapper model—one for the first pass (where `past_key_values` is `None`) and another for subsequent passes (where `past_key_values` contains tensors).
2. Trace each function separately and manage `past_key_values` in the application code.

### Revised Approach

1. **Define Two Separate Functions in Python**:
   - One function for the initial pass (`past_key_values=None`).
   - Another function for subsequent passes with `past_key_values`.

2. **Trace Each Function Separately**:
   - Trace `forward_no_cache` for the first pass.
   - Trace `forward_with_cache` for subsequent passes.

Here’s the updated Python code:

```python
import torch
from transformers import GPT2LMHeadModel

class GPT2WithCache(torch.nn.Module):
    def __init__(self, model):
        super(GPT2WithCache, self).__init__()
        self.model = model

    def forward_no_cache(self, input_ids):
        # First pass with no cache
        outputs = self.model(input_ids, use_cache=True)
        return outputs.logits, outputs.past_key_values

    def forward_with_cache(self, input_ids, past_key_values):
        # Subsequent passes with cache
        outputs = self.model(input_ids, past_key_values=past_key_values, use_cache=True)
        return outputs.logits, outputs.past_key_values

# Initialize the model
model = GPT2LMHeadModel.from_pretrained("gpt2")
wrapped_model = GPT2WithCache(model)

# Define dummy inputs
dummy_input = torch.randint(0, 50257, (1, 1))  # Single token input
dummy_past_key_values = tuple([torch.zeros((1, 12, 0, 64)) for _ in range(24)])  # Example shape for GPT-2

# Trace both functions separately
scripted_model_no_cache = torch.jit.trace(wrapped_model.forward_no_cache, (dummy_input,))
scripted_model_with_cache = torch.jit.trace(wrapped_model.forward_with_cache, (dummy_input, dummy_past_key_values))

# Save both traced models
scripted_model_no_cache.save("gpt2_no_cache.pt")
scripted_model_with_cache.save("gpt2_with_cache.pt")
```

### Step 2: Using the Traced Models in LibTorch

In your C++ application, you can load both traced models and handle the `past_key_values` yourself, calling the appropriate model based on whether it's the first pass or a subsequent pass.

```cpp
#include <torch/script.h>
#include <iostream>

int main() {
    // Load both models
    torch::jit::script::Module model_no_cache;
    torch::jit::script::Module model_with_cache;
    try {
        model_no_cache = torch::jit::load("gpt2_no_cache.pt");
        model_with_cache = torch::jit::load("gpt2_with_cache.pt");
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }

    // Prepare input tensor for the first pass
    torch::Tensor input_ids = torch::randint(0, 50257, {1, 1}, torch::kLong);

    // Initialize past_key_values to None (empty IValue for the first pass)
    torch::jit::IValue past_key_values = torch::jit::IValue();

    // Choose the appropriate model for the first pass
    torch::Tensor logits;
    if (past_key_values.isNone()) {
        // First pass with no cache
        auto output = model_no_cache.forward({input_ids}).toTuple();
        logits = output->elements()[0].toTensor();
        past_key_values = output->elements()[1];  // Update past_key_values with new cache
    }

    // Subsequent passes
    for (int i = 0; i < 20; i++) {
        // Choose the model with cache
        auto output = model_with_cache.forward({input_ids, past_key_values}).toTuple();
        logits = output->elements()[0].toTensor();
        past_key_values = output->elements()[1];  // Update with new cache

        // Select the next token (greedy decoding)
        auto next_token = logits.argmax(-1);

        // Update input_ids for the next pass
        input_ids = next_token.unsqueeze(0);
        std::cout << "Generated token " << i << ": " << next_token.item<int>() << std::endl;
    }

    return 0;
}
```

### Explanation
1. **Two Traced Models**: By splitting `forward_no_cache` and `forward_with_cache`, we avoid any `None` issues in `torch.jit.trace`.
2. **First Pass**: Use `model_no_cache` for the first pass (where `past_key_values` is `None`).
3. **Subsequent Passes**: Use `model_with_cache` and pass `past_key_values` to reuse cached computations.
4. **Updating `input_ids`**: After generating each token, update `input_ids` for the next token generation.

### Summary
This approach effectively bypasses the limitations of `None` handling in `torch.jit.trace` by:
- Splitting the model logic into two functions: one for the initial pass and one for cached passes.
- Tracing each function separately and managing the `past_key_values` state in the application code. 

This should avoid errors while enabling efficient generation with cached states in LibTorch.