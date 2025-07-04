# BF16 config
#HL_FP8=1 \
#HL_TRANSFORMER_IMPL=transformer_engine \
#HL_SEQ_PARALLEL=0 \
#HL_LOAD_DIR=/lkk/Megatron-LM/bigscience/data/oscar/Llama-3.1-8B-Instruct-mcore \
# HL_USE_LAZY_MODE=0 \
HL_USE_DIST_CKPT=1 \
HL_RESULTS_DIR_PREFIX=/scratch-1/lkk \
HL_VERIFY_CKPT=0 \
HL_TRANSFORMER_IMPL=transformer_engine \
HL_SEQ_PARALLEL=0 \
HL_LOAD_DIR=/scratch-1/lkk/Qwen3-30B-A3B/Qwen3-A3B-to-mcore-t8-p1 \
HL_DATA_DIR_ROOT=/scratch-1/lkk/tokenized_Sky-T1_4k_qwen \
HL_DATA_FILE_PREFIX=tokenized_Sky-T1_data_17k_text_document \
HL_TOKENIZER_MODEL=Qwen/Qwen3-30B-A3B \
HL_TOKENIZER_TYPE=HuggingFaceTokenizer \
HL_CKP_ACT=2 \
HL_LLAMA_VER=3 \
HL_DEVICES_PER_NODE=4 \
HL_GBS=8 \
HL_MICRO_BATCH=1 \
HL_LOG_INTERVAL=1 \
HL_SAVE_INTERVAL=5000 \
HL_SEQ_LEN=4096 \
HL_TRAIN_ITERS=2000 \
HL_DP=1 \
HL_TP=1 \
HL_PP=1 \
HL_LLAMA_MODEL_SIZE=A3B \
/sdp/lkk/Habana-Megatron-LM/Megatron-LM-old/examples/llama/sft_qwen3.sh
