import habana_frameworks.torch.core as htcore
import torch
device = torch.device("hpu")

from optimum.habana.checkpoint_utils import (
    get_ds_injection_policy,
    get_repo_root,
    model_is_optimized,
    model_on_meta,
    write_checkpoints_json,
)
from optimum.habana.utils import (
    check_habana_frameworks_version,
    check_optimum_habana_min_version,
    get_habana_frameworks_version,
    set_seed,
)
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

adapt_transformers_to_gaudi()

import argparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Set up argument parsing
parser = argparse.ArgumentParser(description="Script for text generation with a specific model and prompt.")
parser.add_argument('--prompt', type=str, required=True, help="Prompt text to use for text generation")
parser.add_argument('--model-path', type=str, required=True, help="Path to the Huggingface model checkpoint")

# Parse command-line arguments
args = parser.parse_args()

model_path = args.model_path
prompt = args.prompt

config = AutoConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, config=config)
model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16)
print(model)

model = model.eval().to(device)


"""
inputs = tokenizer(prompt, return_tensors="pt")
print(inputs)
print(tokenizer(prompt))
for key in inputs:
    inputs[key] = inputs[key].to(device)
print(inputs)
"""

messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
print(text)

model_inputs = tokenizer([text], return_tensors="pt").to(device)

print(model_inputs)


# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=64
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)

exit()
# top_k, top_p and do_sample are set for greedy argmax based sampling

#outputs = model.generate(**inputs, max_length=100, do_sample=False, top_p=0, top_k=0, temperature=1.0)
# outputs = model.generate(**inputs, max_length=100, do_sample=True, top_p=0.95, top_k=1, temperature=0.1)
# outputs = model.generate(**inputs, max_length=32, do_sample=True, temperature=0.6)
#outputs = model.generate(**inputs, max_length=32, top_k=1)
outputs = model.generate(**inputs, max_length=32, top_p=0.9, temperature=0.6)
print(outputs)
print(outputs[0])
print(tokenizer.decode(outputs[0][len(inputs["input_ids"][0]): ], skip_special_tokens=True))
