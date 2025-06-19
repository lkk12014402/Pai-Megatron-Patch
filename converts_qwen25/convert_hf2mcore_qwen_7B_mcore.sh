cd /lkk/mega_training/commit/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen

bash hf2mcore_qwen2.5_convertor.sh \
7B \
/lkk/Qwen2.5-7B-Instruct/    \
/lkk/Qwen2.5-7B-Instruct/mcore-tp4-pp1  \
4  \
1  \
bf16 \
false \
false
