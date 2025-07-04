cd /scratch-1/lkk/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen

export PT_HPU_GPU_MIGRATION=1

bash hf2mcore_qwen2.5_convertor.sh \
32B \
/scratch-1/lkk/Qwen2.5-32B-Instruct/    \
/scratch-1/lkk/Qwen2.5-32B-Instruct/mcore-tp8-pp1  \
8  \
1  \
bf16 \
false \
false
