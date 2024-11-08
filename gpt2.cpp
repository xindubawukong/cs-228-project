#include "gpt2.h"

#include <iostream>
#include <nlohmann/json.hpp>

#include "gpt2_config.h"
#include "torch/script.h"

using namespace std;

using json = nlohmann::json;

int main() {
  torch::jit::script::Module gpt2_torch =
      torch::jit::load("../model/gpt2_model.pt");

  GPT2Config config = json::parse(ifstream("../model/gpt2_config.json"));

  GPT2 gpt2(config);

  unordered_map<string, at::Tensor> params;
  cout << "parameters:" << endl;
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

  auto logits = output->elements()[0].toTensor();
  cout << "logits: " << logits.sizes() << endl;
  // past_key_values: cached k,v for each layer
  auto past_key_values = output->elements()[1].toTuple();
  cout << "past_key_values: " << past_key_values->size() << endl;
  auto layer0_kv = past_key_values->elements()[0].toTuple();
  // [1, 12, 20, 64] because of multi-head attention
  cout << "layer0_k: " << layer0_kv->elements()[0].toTensor().sizes() << endl;
  cout << "layer0_v: " << layer0_kv->elements()[1].toTensor().sizes() << endl;
  // hidden_states[0-11] are layer outputs, hidden_states[12] is the output of ln_f
  auto hidden_states = output->elements()[2].toTuple();
  cout << "hidden_states: " << hidden_states->size() << endl;
  auto layer0_output = hidden_states->elements()[0].toTensor();
  cout << "layer0_output: " << layer0_output.sizes() << endl;

  return 0;
}
