import torch
import requests
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

device = torch.device("mps")

# model_id = "meta-llama/Llama-3.2-11B-Vision" # 
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id) # , low_cpu_mem_usage=True

# official example
# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# prompt = "<|image|><|begin_of_text|>If I had to write a haiku for this one"
# inputs = processor(image, prompt, return_tensors="pt").to(model.device)

# output = model.generate(**inputs, max_new_tokens=30)
# print(processor.decode(output[0]))


# coco example
img_fp = "../coco_dataset/val2017_task1_horizontal_locate/000000000139.jpg"
image = Image.open(img_fp)

prompt = "<|image|><|begin_of_text|>Can you identify the center of refrigerator is in which block (from 0 to 9). Please just reply one number."
inputs = processor(image, prompt, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=30)
print(processor.decode(output[0]))

