cd /lkk/mega_training/commit/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/llama

bash hf2mcore_convertor_llama3_1.sh \
8B \
/lkk/Megatron-LM/bigscience/data/oscar/out/llama3.1_8b/bf16_transformer_engine_default_nl32_hs4096_ffn14336_gb16_mb1_sp0_D2_T4_P1_devices8_20250508_0709/checkpoints/    \
/lkk/Llama-3.1-8B-Instruct/hf-from-mg  \
4  \
1  \
true \
true \
false \
bf16 \
false \
/lkk/Llama-3.1-8B-Instruct/
