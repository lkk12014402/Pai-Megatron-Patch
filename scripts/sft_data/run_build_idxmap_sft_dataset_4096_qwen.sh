#! /bin/bash

cmd="python build_idxmap_sft_dataset_qwen.py \
  --input /lkk/Sky-T1_data_17k.jsonl \
  --output-prefix tokenized_Sky-T1_4k_qwen/tokenized_Sky-T1_data_17k \
  --patch-tokenizer-type Qwen2Tokenizer \
  --load /lkk/Qwen2.5-7B-Instruct \
  --seq-length 4096 \
  --workers 1 \
  --partitions 1 --debug"

echo $cmd
eval $cmd

