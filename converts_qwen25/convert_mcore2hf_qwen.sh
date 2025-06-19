cd /lkk/mega_training/commit/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen

bash hf2mcore_qwen2.5_convertor.sh \
14B \
/lkk/Megatron-LM/bigscience/data/oscar/out/llama2.5_14b/bf16_local_default_nl48_hs5120_ffn13824_gb4_mb1_sp0_D1_T4_P1_devices4_20250516_1459/checkpoints    \
/lkk/Qwen2.5-14B-Instruct/hf-from-mg  \
4  \
1  \
bf16 \
false \
true \
/lkk/Qwen2.5-14B-Instruct
