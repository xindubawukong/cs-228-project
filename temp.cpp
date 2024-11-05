#include <torch/script.h>  // One-stop header.

#include <iostream>
#include <nlohmann/json.hpp>

using namespace std;

using json = nlohmann::json;

struct GPT2 {
  GPT2(json config) {
    cout << "GPT2 config:" << endl;
    cout << setw(4) << config << endl;
    n_ctx = config["n_ctx"];
    n_embd = config["n_embd"];
    n_head = config["n_head"];
    n_layer = config["n_layer"];
    n_positions = config["n_positions"];
    vocab_size = config["vocab_size"];
    use_cache = config["use_cache"];
  }

  void LoadParameters(auto& named_parameters) {}

  int n_ctx, n_embd, n_head, n_layer, n_positions, vocab_size;
  bool use_cache;
};

int main() {
  torch::jit::script::Module gpt2_torch =
      torch::jit::load("../model/traced_gpt2_model.pt");

  json config = json::parse(ifstream("../model/gpt2_config.json"));

  GPT2 gpt2(config);

  for (const auto& param : gpt2_torch.named_parameters()) {
    std::cout << param.name << ' ' << param.value.sizes() << std::endl;
  }

  // Prepare input
  torch::Tensor input_ids = torch::randint(0, 50257, {1, 20}, torch::kLong);

  // Run the model and get the outputs
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input_ids);

  // Execute the model
  auto output = gpt2_torch.forward(inputs).toTuple();

  auto t = output->elements()[0].toTensor();
  cout << t.sizes() << endl;
  auto x = output->elements()[1].toTuple();
  cout << x->size() << endl;

  return 0;
}
