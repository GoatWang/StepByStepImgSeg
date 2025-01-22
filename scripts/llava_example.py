import torch
import requests
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

device = torch.device("mps")


model_name = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_name)
model = LlavaForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

# Load an image from a URL or local path
img_fp = "../coco_dataset/val2017_task1_horizontal_locate/000000000139.jpg"
# img_fp = "../coco_dataset/val2017_task2_vertical_locate/000000000139.jpg"
image = Image.open(img_fp)

# Define your question
# question = "What is shown in this image?"
object_name = "tv" # refrigerator
question = f"Can you identify the center of {object_name} is in which block (from 0 to 9). Please just reply one number."

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ],
    },
]

# Apply the chat template
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=30)

# Decode the generated response
answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Question:", question)
print("Answer:", answer)
