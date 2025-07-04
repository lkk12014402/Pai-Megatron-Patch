cd /scratch-1/lkk/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen

export PT_HPU_GPU_MIGRATION=1

bash hf2mcore_qwen3_convertor_new.sh \
32B \
/scratch-1/lkk/Qwen3-32B/    \
/scratch-1/lkk/Qwen3-32B/mcore-tp8-pp1  \
8  \
1  \
bf16 \
false \
false
