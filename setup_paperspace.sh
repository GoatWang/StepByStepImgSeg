apt install -y tmux
pip install transformers -U
pip install pycocotools
pip install 'accelerate>=0.26.0'

# install flash attention 2: ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the following error: the package flash_attn seems to be not installed. Please refer to the documentation of https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2 to install Flash Attention 2.
# pip install flash-attn --no-build-isolation

mkdir -p temp
if [ ! -e "/notebooks/StepByStepImgSeg/temp/LLaVA" ]; then
    cd /notebooks/StepByStepImgSeg/temp
    git clone https://github.com/haotian-liu/LLaVA
fi
export PYTHONPATH=/notebooks/StepByStepImgSeg/temp/llava:$PYTHONPATH

