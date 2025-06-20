cd /scratch-1/lkk/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor

export PT_HPU_GPU_MIGRATION=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export KUBERNETES_CONTAINER_RESOURCE_GPU=4

bash scripts/qwen3/run_8xH20.sh \
14B \
/scratch-1/lkk/Qwen3-14B/ \
/scratch-1/lkk/Qwen3-14B/Qwen3-14B-to-mcore-t4-p1  \
false \
true \
bf16
