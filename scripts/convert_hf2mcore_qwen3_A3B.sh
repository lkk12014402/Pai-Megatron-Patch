cd /scratch-1/lkk/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor

export PT_HPU_GPU_MIGRATION=1

#export CUDA_VISIBLE_DEVICES=0,5,6,7
#export KUBERNETES_CONTAINER_RESOURCE_GPU=4

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export KUBERNETES_CONTAINER_RESOURCE_GPU=8

bash scripts/qwen3/run_8xH20.sh \
A3B \
/scratch-1/lkk/Qwen3-30B-A3B/ \
/scratch-1/lkk/Qwen3-30B-A3B/Qwen3-A3B-to-mcore-tp1-pp1-ep8  \
false \
true \
bf16
