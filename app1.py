from PIL import Image
import requests
from transformers import AutoProcessor, BlipForConditionalGeneration
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "A picture of"
inputs = processor(images=image, text=text, return_tensors="pt")
outputs = model.generate(**inputs)
processor.decode(outputs[0], skip_special_token=True)