cd /scratch-1/lkk/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen

export PT_HPU_GPU_MIGRATION=1

bash hf2mcore_qwen2.5_convertor.sh \
14B \
/scratch-1/lkk/Qwen2.5-14B-Instruct/    \
/scratch-1/lkk/Qwen2.5-14B-Instruct/mcore-tp4-pp1-te  \
4  \
1  \
bf16 \
true \
false
