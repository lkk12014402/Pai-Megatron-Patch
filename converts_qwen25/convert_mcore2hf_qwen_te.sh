cd /lkk/mega_training/commit/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen

bash hf2mcore_qwen2.5_convertor.sh \
14B \
/lkk/Megatron-LM/bigscience/data/oscar/out/llama2.5_14b/bf16_gb8_mb2_sp0_D1_T4_P1_cards4/checkpoints    \
/lkk/Qwen2.5-14B-Instruct/hf-from-mg-te  \
4  \
1  \
bf16 \
true \
true \
/lkk/Qwen2.5-14B-Instruct
