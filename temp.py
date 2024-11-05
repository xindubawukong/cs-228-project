import torch
from transformers import GPT2LMHeadModel, GPT2Config
from typing import Dict

class GPT2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super(GPT2Wrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, past_key_values=None):
        output = self.model(input_ids,
                            output_hidden_states=True,
                            past_key_values=past_key_values,
                            use_cache=True
                            )
        return (output.logits, output.past_key_values, output.hidden_states)

# Load the original model and wrap it
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_wrapped = GPT2Wrapper(gpt2)
traced_model = torch.jit.trace(gpt2_wrapped, example_inputs=torch.randint(0, 50257, (1, 20)))
traced_model.save('model/traced_gpt2_model.pt')

gpt2.config.to_json_file('model/gpt2_config.json')








# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import torch

# # Load model and tokenizer
# model_name = "gpt2"
# model = GPT2LMHeadModel.from_pretrained(model_name)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# # Set the model to evaluation mode
# model.eval()

# # Initial prompt to start generation
# prompt = "Once upon a time"
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# # Variables to store generated tokens and past key values
# generated_tokens = input_ids
# past_key_values = None

# # Number of tokens to generate
# max_new_tokens = 20

# for i in range(max_new_tokens):
#     # Only pass the last generated token to the model, along with past_key_values
#     outputs = model(input_ids=generated_tokens[:, -1:], past_key_values=past_key_values, use_cache=True)
    
#     # Get the logits and past_key_values (cache)
#     logits = outputs.logits
#     past_key_values = outputs.past_key_values  # Update cache with new past_key_values

#     # Greedy decoding: pick the token with the highest probability
#     next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

#     # Append the generated token to the sequence
#     generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

#     # Decode and print the generated token
#     generated_text = tokenizer.decode(next_token[0])
#     print(generated_text, end="")

# # Decode the entire generated sequence
# full_generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
# print("\n\nFull generated text:\n", full_generated_text)
