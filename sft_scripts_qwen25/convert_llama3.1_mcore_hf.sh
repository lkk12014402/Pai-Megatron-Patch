export PT_HPU_GPU_MIGRATION=1
python /lkk/Megatron-LM/tools/checkpoint/convert_mlm_to_hf_checkpoint.py \
    --ckpt-dir-name "iter_0000001" \
    --target-params-dtype "bf16" \
    --source-model-type "llama3.1" \
    --load-path "./Llama-3.1-8B-Instruct-mcore" \
    --save-path "sft_sky_llama3.1_8b_instruct_hf_checkpoints/"
