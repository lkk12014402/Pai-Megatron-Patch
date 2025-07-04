
docker run -d -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -v /home/sdp:/sdp -v /scratch-1:/scratch-1 -e https_proxy=$https_proxy -e http_proxy=$http_proxy -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host --shm-size=10g --name lkk-llm vault.habana.ai/gaudi-docker/1.21.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest bash



安装git-lfs
https://zhuanlan.zhihu.com/p/545762616


pip install --upgrade huggingface_hub
huggingface-cli login



export MEGATRON_LM_ROOT=/sdp/lkk/Habana-Megatron-LM/Megatron-LM
export MEGATRON_PATCH_LM_ROOT=/scratch-1/lkk/Pai-Megatron-Patch
export PYTHONPATH=$MEGATRON_LM_ROOT:$MEGATRON_PATCH_LM_ROOT:$PYTHONPATH

pip install transformers
pip install accelerate
pip install datasets

For running qwen2.5

export MEGATRON_LM_ROOT=/sdp/lkk/Habana-Megatron-LM/Megatron-LM-old
export MEGATRON_PATCH_LM_ROOT=/scratch-1/lkk/Pai-Megatron-Patch
export PYTHONPATH=$MEGATRON_LM_ROOT:$MEGATRON_PATCH_LM_ROOT:$PYTHONPATH


export MEGATRON_LM_ROOT=/sdp/lkk/mega_training/Pai-Megatron-Patch/Habana-Megatron-LM/Megatron-LM
export MEGATRON_PATCH_LM_ROOT=/sdp/lkk/mega_training/Pai-Megatron-Patch
export PYTHONPATH=$MEGATRON_LM_ROOT:$MEGATRON_PATCH_LM_ROOT:$PYTHONPATH





export MEGATRON_LM_ROOT=/sdp/lkk/mega_training/Pai-Megatron-Patch/Habana-Megatron-LM/Megatron-LM-old
export MEGATRON_PATCH_LM_ROOT=/scratch-1/lkk/Pai-Megatron-Patch
export PYTHONPATH=$MEGATRON_LM_ROOT:$MEGATRON_PATCH_LM_ROOT:$PYTHONPATH




Convert:

export MEGATRON_LM_ROOT=/sdp/lkk/mega_training/Pai-Megatron-Patch/Habana-Megatron-LM/Megatron-LM-old
export MEGATRON_PATCH_LM_ROOT=/scratch-1/lkk/Pai-Megatron-Patch
export PYTHONPATH=$MEGATRON_LM_ROOT:$MEGATRON_PATCH_LM_ROOT:$PYTHONPATH

/scratch-1/lkk/Pai-Megatron-Patch/scripts/convert_hf2mcore_qwen3_8B_mcore_torch.sh


Training:

export MEGATRON_LM_ROOT=/sdp/lkk/Habana-Megatron-LM/Megatron-LM
export MEGATRON_LM_ROOT=/sdp/lkk/mega_training/Pai-Megatron-Patch/Habana-Megatron-LM/Megatron-LM
export MEGATRON_PATCH_LM_ROOT=/scratch-1/lkk/Pai-Megatron-Patch
export PYTHONPATH=$MEGATRON_LM_ROOT:$MEGATRON_PATCH_LM_ROOT:$PYTHONPATH


/sdp/lkk/Habana-Megatron-LM/Megatron-LM-old/bigscience/data/oscar/debug_sft_qwen3_8B_torch_te.sh



export MEGATRON_LM_ROOT=/sdp/lkk/mega_training/Pai-Megatron-Patch/Habana-Megatron-LM/Megatron-LM-fp4
export MEGATRON_PATCH_LM_ROOT=/scratch-1/lkk/Pai-Megatron-Patch
export PYTHONPATH=$MEGATRON_LM_ROOT:$MEGATRON_PATCH_LM_ROOT:$PYTHONPATH


/sdp/lkk/Habana-Megatron-LM/Megatron-LM-old/bigscience/data/oscar/debug_sft_qwen3_8B_torch_te.sh


debug_sft_qwen2.5_7b_te_fp8.sh
/sdp/lkk/Habana-Megatron-LM/Megatron-LM-old/examples/llama/sft_qwen.sh



nohup bash run_all_no_distributed_optimizer_fp4.sh  >> run_no_distributed_optimizer_fp4.log 2>&1 &


Moe

export MEGATRON_LM_ROOT=/sdp/lkk/Pai-Megatron-Patch/Qwen3moe_Megatron-LM
export MEGATRON_PATCH_LM_ROOT=/sdp/lkk/Pai-Megatron-Patch/
export PYTHONPATH=$MEGATRON_LM_ROOT:$MEGATRON_PATCH_LM_ROOT:$PYTHONPATH



/sdp/lkk/Habana-Megatron-LM/Megatron-LM-old/bigscience/data/oscar/debug_sft_qwen3_A3B_te.sh


