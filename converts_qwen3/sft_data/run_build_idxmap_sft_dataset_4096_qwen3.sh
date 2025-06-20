#! /bin/bash

cmd="python build_idxmap_sft_dataset_qwen.py \
  --input /scratch-2/lkk/Sky-T1_data_17k.jsonl \
  --output-prefix /scratch-2/lkk/tokenized_Sky-T1_4k_qwen3/tokenized_Sky-T1_data_17k \
  --patch-tokenizer-type Qwen3Tokenizer \
  --load /scratch-2/lkk/Qwen3-8B \
  --seq-length 4096 \
  --workers 1 \
  --partitions 1 --debug"

echo $cmd
eval $cmd

