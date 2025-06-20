cd /lkk/mega_training/commit/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/llama

bash hf2mcore_convertor_llama3_1.sh \
8B \
/lkk/Llama-3.1-8B-Instruct/    \
/lkk/Llama-3.1-8B-Instruct/mcore-tp4-pp1-te  \
4  \
1  \
false \
true \
false \
bf16 \
true
