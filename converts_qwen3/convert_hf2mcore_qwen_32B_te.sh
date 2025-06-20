cd /scratch-2/lkk/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen

bash hf2mcore_qwen2.5_convertor.sh \
32B \
/scratch-2/lkk/Qwen2.5-32B-Instruct/    \
/scratch-2/lkk/Qwen2.5-32B-Instruct/mcore-tp1-pp8-te  \
1  \
8  \
bf16 \
true \
false
