cd /scratch-1/lkk/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen

export PT_HPU_GPU_MIGRATION=1

bash hf2mcore_qwen3_convertor_new.sh \
0.6B \
/scratch-1/lkk/Qwen3-0.6B/    \
/scratch-1/lkk/Qwen3-0.6B/mcore-tp1-pp1-te  \
1  \
1  \
bf16 \
true \
false
