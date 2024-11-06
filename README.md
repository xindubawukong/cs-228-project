
## Usage

### Download the Repo

```
git clone --recurse-submodules git@github.com:xindubawukong/cs-228-project.git
```

### Download libtorch

Download from https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.5.1%2Bcpu.zip

Put libtorch at `third_party/libtorch`

### Save GPT2 Model

```
python3 get_gpt2_model.py
```

This will save parameters to `model/gpt2_model.pt` and config to `model/gpt2_config.json`.

### Run GPT2 in C++

```
mkdir -p build && cd build
cmake ..
make
./gpt2
```

It will read model parameters from the saved files.



```
GPT2 config:
{
    "_attn_implementation_autoset": true,
    "_name_or_path": "gpt2",
    "activation_function": "gelu_new",
    "architectures": [
        "GPT2LMHeadModel"
    ],
    "attn_pdrop": 0.1,
    "bos_token_id": 50256,
    "embd_pdrop": 0.1,
    "eos_token_id": 50256,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "gpt2",
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_inner": null,
    "n_layer": 12,
    "n_positions": 1024,
    "reorder_and_upcast_attn": false,
    "resid_pdrop": 0.1,
    "scale_attn_by_inverse_layer_idx": false,
    "scale_attn_weights": true,
    "summary_activation": null,
    "summary_first_dropout": 0.1,
    "summary_proj_to_labels": true,
    "summary_type": "cls_index",
    "summary_use_proj": true,
    "task_specific_params": {
        "text-generation": {
            "do_sample": true,
            "max_length": 50
        }
    },
    "transformers_version": "4.46.1",
    "use_cache": true,
    "vocab_size": 50257
}

parameters:
model.transformer.wte.weight [50257, 768]
model.transformer.wpe.weight [1024, 768]
model.transformer.h.0.ln_1.weight [768]
model.transformer.h.0.ln_1.bias [768]
model.transformer.h.0.attn.c_attn.weight [768, 2304]
model.transformer.h.0.attn.c_attn.bias [2304]
model.transformer.h.0.attn.c_proj.weight [768, 768]
model.transformer.h.0.attn.c_proj.bias [768]
model.transformer.h.0.ln_2.weight [768]
model.transformer.h.0.ln_2.bias [768]
model.transformer.h.0.mlp.c_fc.weight [768, 3072]
model.transformer.h.0.mlp.c_fc.bias [3072]
model.transformer.h.0.mlp.c_proj.weight [3072, 768]
model.transformer.h.0.mlp.c_proj.bias [768]
model.transformer.h.1.ln_1.weight [768]
model.transformer.h.1.ln_1.bias [768]
model.transformer.h.1.attn.c_attn.weight [768, 2304]
model.transformer.h.1.attn.c_attn.bias [2304]
model.transformer.h.1.attn.c_proj.weight [768, 768]
model.transformer.h.1.attn.c_proj.bias [768]
model.transformer.h.1.ln_2.weight [768]
model.transformer.h.1.ln_2.bias [768]
model.transformer.h.1.mlp.c_fc.weight [768, 3072]
model.transformer.h.1.mlp.c_fc.bias [3072]
model.transformer.h.1.mlp.c_proj.weight [3072, 768]
model.transformer.h.1.mlp.c_proj.bias [768]
model.transformer.h.2.ln_1.weight [768]
model.transformer.h.2.ln_1.bias [768]
model.transformer.h.2.attn.c_attn.weight [768, 2304]
model.transformer.h.2.attn.c_attn.bias [2304]
model.transformer.h.2.attn.c_proj.weight [768, 768]
model.transformer.h.2.attn.c_proj.bias [768]
model.transformer.h.2.ln_2.weight [768]
model.transformer.h.2.ln_2.bias [768]
model.transformer.h.2.mlp.c_fc.weight [768, 3072]
model.transformer.h.2.mlp.c_fc.bias [3072]
model.transformer.h.2.mlp.c_proj.weight [3072, 768]
model.transformer.h.2.mlp.c_proj.bias [768]
model.transformer.h.3.ln_1.weight [768]
model.transformer.h.3.ln_1.bias [768]
model.transformer.h.3.attn.c_attn.weight [768, 2304]
model.transformer.h.3.attn.c_attn.bias [2304]
model.transformer.h.3.attn.c_proj.weight [768, 768]
model.transformer.h.3.attn.c_proj.bias [768]
model.transformer.h.3.ln_2.weight [768]
model.transformer.h.3.ln_2.bias [768]
model.transformer.h.3.mlp.c_fc.weight [768, 3072]
model.transformer.h.3.mlp.c_fc.bias [3072]
model.transformer.h.3.mlp.c_proj.weight [3072, 768]
model.transformer.h.3.mlp.c_proj.bias [768]
model.transformer.h.4.ln_1.weight [768]
model.transformer.h.4.ln_1.bias [768]
model.transformer.h.4.attn.c_attn.weight [768, 2304]
model.transformer.h.4.attn.c_attn.bias [2304]
model.transformer.h.4.attn.c_proj.weight [768, 768]
model.transformer.h.4.attn.c_proj.bias [768]
model.transformer.h.4.ln_2.weight [768]
model.transformer.h.4.ln_2.bias [768]
model.transformer.h.4.mlp.c_fc.weight [768, 3072]
model.transformer.h.4.mlp.c_fc.bias [3072]
model.transformer.h.4.mlp.c_proj.weight [3072, 768]
model.transformer.h.4.mlp.c_proj.bias [768]
model.transformer.h.5.ln_1.weight [768]
model.transformer.h.5.ln_1.bias [768]
model.transformer.h.5.attn.c_attn.weight [768, 2304]
model.transformer.h.5.attn.c_attn.bias [2304]
model.transformer.h.5.attn.c_proj.weight [768, 768]
model.transformer.h.5.attn.c_proj.bias [768]
model.transformer.h.5.ln_2.weight [768]
model.transformer.h.5.ln_2.bias [768]
model.transformer.h.5.mlp.c_fc.weight [768, 3072]
model.transformer.h.5.mlp.c_fc.bias [3072]
model.transformer.h.5.mlp.c_proj.weight [3072, 768]
model.transformer.h.5.mlp.c_proj.bias [768]
model.transformer.h.6.ln_1.weight [768]
model.transformer.h.6.ln_1.bias [768]
model.transformer.h.6.attn.c_attn.weight [768, 2304]
model.transformer.h.6.attn.c_attn.bias [2304]
model.transformer.h.6.attn.c_proj.weight [768, 768]
model.transformer.h.6.attn.c_proj.bias [768]
model.transformer.h.6.ln_2.weight [768]
model.transformer.h.6.ln_2.bias [768]
model.transformer.h.6.mlp.c_fc.weight [768, 3072]
model.transformer.h.6.mlp.c_fc.bias [3072]
model.transformer.h.6.mlp.c_proj.weight [3072, 768]
model.transformer.h.6.mlp.c_proj.bias [768]
model.transformer.h.7.ln_1.weight [768]
model.transformer.h.7.ln_1.bias [768]
model.transformer.h.7.attn.c_attn.weight [768, 2304]
model.transformer.h.7.attn.c_attn.bias [2304]
model.transformer.h.7.attn.c_proj.weight [768, 768]
model.transformer.h.7.attn.c_proj.bias [768]
model.transformer.h.7.ln_2.weight [768]
model.transformer.h.7.ln_2.bias [768]
model.transformer.h.7.mlp.c_fc.weight [768, 3072]
model.transformer.h.7.mlp.c_fc.bias [3072]
model.transformer.h.7.mlp.c_proj.weight [3072, 768]
model.transformer.h.7.mlp.c_proj.bias [768]
model.transformer.h.8.ln_1.weight [768]
model.transformer.h.8.ln_1.bias [768]
model.transformer.h.8.attn.c_attn.weight [768, 2304]
model.transformer.h.8.attn.c_attn.bias [2304]
model.transformer.h.8.attn.c_proj.weight [768, 768]
model.transformer.h.8.attn.c_proj.bias [768]
model.transformer.h.8.ln_2.weight [768]
model.transformer.h.8.ln_2.bias [768]
model.transformer.h.8.mlp.c_fc.weight [768, 3072]
model.transformer.h.8.mlp.c_fc.bias [3072]
model.transformer.h.8.mlp.c_proj.weight [3072, 768]
model.transformer.h.8.mlp.c_proj.bias [768]
model.transformer.h.9.ln_1.weight [768]
model.transformer.h.9.ln_1.bias [768]
model.transformer.h.9.attn.c_attn.weight [768, 2304]
model.transformer.h.9.attn.c_attn.bias [2304]
model.transformer.h.9.attn.c_proj.weight [768, 768]
model.transformer.h.9.attn.c_proj.bias [768]
model.transformer.h.9.ln_2.weight [768]
model.transformer.h.9.ln_2.bias [768]
model.transformer.h.9.mlp.c_fc.weight [768, 3072]
model.transformer.h.9.mlp.c_fc.bias [3072]
model.transformer.h.9.mlp.c_proj.weight [3072, 768]
model.transformer.h.9.mlp.c_proj.bias [768]
model.transformer.h.10.ln_1.weight [768]
model.transformer.h.10.ln_1.bias [768]
model.transformer.h.10.attn.c_attn.weight [768, 2304]
model.transformer.h.10.attn.c_attn.bias [2304]
model.transformer.h.10.attn.c_proj.weight [768, 768]
model.transformer.h.10.attn.c_proj.bias [768]
model.transformer.h.10.ln_2.weight [768]
model.transformer.h.10.ln_2.bias [768]
model.transformer.h.10.mlp.c_fc.weight [768, 3072]
model.transformer.h.10.mlp.c_fc.bias [3072]
model.transformer.h.10.mlp.c_proj.weight [3072, 768]
model.transformer.h.10.mlp.c_proj.bias [768]
model.transformer.h.11.ln_1.weight [768]
model.transformer.h.11.ln_1.bias [768]
model.transformer.h.11.attn.c_attn.weight [768, 2304]
model.transformer.h.11.attn.c_attn.bias [2304]
model.transformer.h.11.attn.c_proj.weight [768, 768]
model.transformer.h.11.attn.c_proj.bias [768]
model.transformer.h.11.ln_2.weight [768]
model.transformer.h.11.ln_2.bias [768]
model.transformer.h.11.mlp.c_fc.weight [768, 3072]
model.transformer.h.11.mlp.c_fc.bias [3072]
model.transformer.h.11.mlp.c_proj.weight [3072, 768]
model.transformer.h.11.mlp.c_proj.bias [768]
model.transformer.ln_f.weight [768]
model.transformer.ln_f.bias [768]
model.lm_head.weight [50257, 768]
```


<!-- 

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

This should avoid errors while enabling efficient generation with cached states in LibTorch. -->