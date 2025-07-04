cd /scratch-1/lkk/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen

export PT_HPU_GPU_MIGRATION=1

python te2inc.py /scratch-1/lkk/Qwen3-0.6B/hf-from-mg-te-fp8 /scratch-1/lkk/Qwen3-0.6B/hf-from-mg-te-fp8-vllm-static

INC_VLLM_DYN=1 python te2inc.py /scratch-1/lkk/Qwen3-0.6B/hf-from-mg-te-fp8 /scratch-1/lkk/Qwen3-0.6B/hf-from-mg-te-fp8-vllm-dynamic
