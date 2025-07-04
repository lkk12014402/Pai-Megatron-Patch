cd /scratch-1/lkk/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen

export PT_HPU_GPU_MIGRATION=1

bash hf2mcore_qwen3_convertor_new.sh \
0.6B \
/scratch-1/lkk/out/llama3_0.6b/bf16_transformer_engine_default_nl28_hs1024_ffn3072_gb8_mb1_sp0_D8_T1_P1_devices8_20250626_1206/checkpoints/  \
/scratch-1/lkk/Qwen3-0.6B/hf-from-mg-bf16sft \
1  \
1  \
bf16 \
true \
true \
/scratch-1/lkk/Qwen3-0.6B/
