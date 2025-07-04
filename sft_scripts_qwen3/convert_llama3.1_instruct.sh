export PT_HPU_GPU_MIGRATION=1
python /lkk/Megatron-LM/tools/checkpoint/convert.py \
    --bf16 \
    --model-type GPT \
    --loader llama_mistral \
    --saver mcore \
    --target-tensor-parallel-size 1 \
    --checkpoint-type hf \
    --load-dir /lkk/Llama-3.1-8B-Instruct/ \
    --save-dir "Llama-3.1-8B-Instruct-mcore" \
    --tokenizer-model /lkk/Llama-3.1-8B/original/tokenizer.model \
    --model-size llama3-8B \
    --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 1 \
    --true-vocab-size 128256 \
    --bf16
