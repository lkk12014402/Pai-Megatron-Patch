cd /lkk/mega_training/commit/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen

bash hf2mcore_qwen2.5_convertor.sh \
32B \
/lkk/Qwen2.5-32B-Instruct/    \
/lkk/Qwen2.5-32B-Instruct/mcore-tp1-pp8-te  \
1  \
8  \
bf16 \
true \
false
