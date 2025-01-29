pip3 install opencv-python
pip3 install pycocotools
pip3 install pillow
pip3 install transformers
pip3 install 'accelerate>=0.26.0'
pip3 install deepspeed
pip3 install peft
pip3 install sentencepiece
pip3 install protobuf
pip3 install wandb

wandb login 427974ce1dec11546ede262db4206a90fcf9ce00

export PYTHONPATH=/notebooks/StepByStepImgSeg/temp/llava:$PYTHONPATH
echo "export PYTHONPATH=/notebooks/StepByStepImgSeg/temp/llava:$PYTHONPATH" >> ~/.bashrc

huggingface-cli download liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5 --local-dir /root/StepByStepImgSeg/temp/LLaVA/checkpoints/vicuna-7b-v1.5-pretrain