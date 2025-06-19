cd /lkk/mega_training/commit/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor

export CUDA_VISIBLE_DEVICES=2,4,5,6
export KUBERNETES_CONTAINER_RESOURCE_GPU=4

bash scripts/qwen3/run_8xH20.sh \
8B \
/lkk/Qwen3-8B \
/lkk/Qwen3-8B/Qwen3-8B-to-mcore  \
false \
true \
bf16
