
set -ex

export PT_HPU_GPU_MIGRATION=1
export PT_HPU_WEIGHT_SHARING=0
export VLLM_SKIP_WARMUP=true
export VLLM_DELAYED_SAMPLING=true


lm_eval --model vllm \
   --model_args pretrained=$1,tensor_parallel_size=1,data_parallel_size=1 \
   --tasks lambada_openai \
   --batch_size 8 
