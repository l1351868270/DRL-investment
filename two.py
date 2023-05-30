from transformers import AutoImageProcessor, ResNetModel
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"]

# image_processor = AutoImageProcessor.from_pretrained("./model")
# model = ResNetModel.from_pretrained("./model")

image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetModel.from_pretrained("microsoft/resnet-50")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    for i in range(1000000):
        outputs = model(**inputs)
        if i % 1000 == 0:
            print()


# last_hidden_states = outputs.last_hidden_state
# list(last_hidden_states.shape)
# [1, 2048, 7, 7]