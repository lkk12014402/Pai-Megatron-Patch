cd /lkk/mega_training/commit/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen

bash hf2mcore_qwen2.5_convertor.sh \
14B \
/lkk/Qwen2.5-14B-Instruct/    \
/lkk/Qwen2.5-14B-Instruct/mcore-tp4-pp1-te  \
4  \
1  \
bf16 \
true \
false
