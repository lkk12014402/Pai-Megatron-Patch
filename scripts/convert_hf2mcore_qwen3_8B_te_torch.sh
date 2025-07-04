cd /scratch-1/lkk/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen

export PT_HPU_GPU_MIGRATION=1

bash hf2mcore_qwen3_convertor_new.sh \
8B \
/scratch-1/lkk/Qwen3-8B/    \
/scratch-1/lkk/Qwen3-8B/mcore-tp4-pp1-te  \
4  \
1  \
bf16 \
true \
false
