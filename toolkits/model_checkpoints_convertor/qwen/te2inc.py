def create_te2inc_fp8_config(fp8_meta_dict, name_mapping):
    """Convert transformer_engine fp8 meta into INC qconfig.

    Args:
        fp8_meta_dict (dict): fp8 meta information

    Returns:
        qconfig: INC FP8Config object
    """
    def add_node(v, op_name, node_info):
        index = v['amax_history_index_fwd']
        input_amax = v['amax_history_fwd'][:, 0].amax()
        weight_amax = v['amax_history_fwd'][index.item()][1]
        node_info[op_name] = {"inputs": [[[input_amax.item()]]], "params": {"weight": [[weight_amax.item()]]}}
        return node_info

    import numpy as np
    import json
    from neural_compressor.torch.quantization import FP8Config
    measurement_info = {}
    measurement_info["GlobalRank"] = None
    measurement_info["LocalRank"] = None
    measurement_info["Mode"] = "DynamicRange"
    # collect node info
    node_info = {}
    for k, v in fp8_meta_dict.items():
        if v is None:
            continue
        op_name = k.rstrip("." + torch.nn.modules.module._EXTRA_STATE_KEY_SUFFIX)
        for te_name, hf_name in name_mapping.items():
            if te_name not in op_name:
                continue
            if isinstance(hf_name, (list, tuple)):
                for name in hf_name:
                    new_name = op_name.replace(te_name, name)
                    node_info = add_node(v, new_name, node_info)
            else:
                op_name = op_name.replace(te_name, hf_name)


    fp8_config = "E5M2" if v['extra_fp8_variables']['fp8_max_fwd'] == 57344 else "E4M3"
    fp8_config = "E5M2"
    measurement_info["Nodes"] = node_info
    # create hqt_output files
    os.makedirs("hqt_output", exist_ok=True)
    np.savez("hqt_output/measure_hooks_maxabs.npz", measurement_info)
    with open("hqt_output/measure_hooks_maxabs.json", "w", encoding="utf-8") as file:
        json.dump(measurement_info, file, indent=4)
    with open("hqt_output/measure_hooks_maxabs_mod_list.json", "w", encoding="utf-8") as file:
        json.dump(list(measurement_info["Nodes"].keys()), file, indent=4)
    print("The fp8 information required by INC is now saved in the hqt_output folder.")
    # create qconfig which can leverage hqt_output to execute fp8 quantization
    qconfig = FP8Config(fp8_config=fp8_config, allowlist={"names": list(node_info.keys())})
    return qconfig


import torch
import transformers
import sys
import os

# create hf model
hf_ckpt_path = sys.argv[1]
save_path = sys.argv[2]
config = transformers.AutoConfig.from_pretrained(hf_ckpt_path)
tokenizer = transformers.AutoTokenizer.from_pretrained(hf_ckpt_path)
hf_model = transformers.AutoModelForCausalLM.from_pretrained(hf_ckpt_path, torch_dtype=config.torch_dtype)

# te_name: hf_name
name_mapping = {
    # model level
    "decoder.layers": "model.layers",
    # op level, please use list for value to identify with model level
    "self_attention.linear_qkv": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    "self_attention.linear_proj": ["self_attn.o_proj"],
    "mlp.linear_fc1": ["mlp.gate_proj", "mlp.up_proj"],
    "mlp.linear_fc2": ["mlp.down_proj"]
}

# create fp8 config and measurement files in hqt_output.
fp8_meta = torch.load(os.path.join(hf_ckpt_path, "fp8_meta.pt"))
fp8_config = create_te2inc_fp8_config(fp8_meta, name_mapping=name_mapping)
# create fp8 model
from neural_compressor.torch.quantization import convert, save
fp8_model = convert(hf_model, fp8_config)

save(fp8_model, checkpoint_dir=save_path, format="vllm")
tokenizer.save_pretrained(save_path)
exit()

from neural_compressor.evaluation.lm_eval import evaluate, LMEvalParser
eval_args = LMEvalParser(
    model="hf",
    user_model=fp8_model,
    tokenizer=tokenizer,
    batch_size=1,
    tasks='lambada_openai',
    device="hpu",
    pad_to_buckets=True,
    num_fewshot=0,
    limit=10,
)                      
results = evaluate(eval_args)
torch.hpu.synchronize()


save(fp8_model, checkpoint_dir="saved_results", format="vllm")

# python te2inc.py /scratch-2/xinhe/Qwen3-0.6B/kaokao-hf

