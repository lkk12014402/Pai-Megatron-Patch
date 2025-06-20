# BF16 config
#HL_FP8=1 \
#HL_TRANSFORMER_IMPL=transformer_engine \
#HL_SEQ_PARALLEL=0 \
#HL_LOAD_DIR=/lkk/Megatron-LM/bigscience/data/oscar/Llama-3.1-8B-Instruct-mcore \

# fp4 params
export WEIGHT_FP4="true"
export INPUT_FP4="true"
export GRAD_FP4="true"
export WEIGHT_GRID="true"
export INPUT_SR="true" # BWD only
export GRAD_SR="true" # BWD only
export WEIGHT_FT="false"
export BLOCK_SIZE=16
export SCALE_FORMAT="e4m3" # "e8m0", "e4m3", "ue5m3", "bf16"
export BIAS_FP4="true"

export SCALED_SWIGLU="false"
export DETACH_SCALED_SWIGLU="true"
# export DELAYED_SCALED_SWIGLU="true"
export FULL_FP4="true"


HL_USE_DISTRIBUTED_OPTIMIZER=0 \
HL_VERIFY_CKPT=0 \
HL_TRANSFORMER_IMPL=local \
HL_SEQ_PARALLEL=0 \
HL_LOAD_DIR=/lkk/Qwen2.5-7B-Instruct/mcore-tp4-pp1 \
HL_DATA_DIR_ROOT=/lkk/mega_training/commit/Pai-Megatron-Patch/scripts/sft_data/tokenized_Sky-T1_4k_qwen \
HL_DATA_FILE_PREFIX=tokenized_Sky-T1_data_17k_text_document \
HL_TOKENIZER_MODEL=/lkk/Llama-3.1-8B-Instruct/original/tokenizer.model \
HL_TOKENIZER_TYPE=Llama3Tokenizer \
HL_CKP_ACT=2 \
HL_LLAMA_VER=2.5 \
HL_DEVICES_PER_NODE=8 \
HL_GBS=8 \
HL_MICRO_BATCH=1 \
HL_LOG_INTERVAL=1 \
HL_SAVE_INTERVAL=5000 \
HL_SEQ_LEN=4096 \
HL_TRAIN_ITERS=2000 \
HL_DP=2 \
HL_TP=4 \
HL_PP=1 \
HL_LLAMA_MODEL_SIZE=7 \
/lkk/Megatron-LM/examples/llama/sft_qwen.sh
