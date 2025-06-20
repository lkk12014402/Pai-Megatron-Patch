cd /scratch-2/lkk/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen

bash hf2mcore_qwen2.5_convertor.sh \
72B \
/scratch-2/lkk/Qwen2.5-72B-Instruct/    \
/scratch-2/lkk/Qwen2.5-72B-Instruct/mcore-tp8-pp1-te  \
8  \
1  \
bf16 \
true \
false
