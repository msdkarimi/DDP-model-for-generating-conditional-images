from transformers import CLIPTokenizer, CLIPTextModel
import os
model_name = "openai/clip-vit-large-patch14"
save_path = "./clip_model"  # Change this to your preferred directory
os.makedirs(save_path, exist_ok=True)
tokenizer = CLIPTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_path)

model = CLIPTextModel.from_pretrained(model_name)
model.save_pretrained(save_path)
