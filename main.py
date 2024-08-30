import streamlit as st
from PIL import Image
import requests
from transformers import AutoProcessor, BlipForConditionalGeneration

def generate_caption(image_url, model_name="Salesforce/blip-image-captioning-base"):
    """Fetches an image, generates a caption using BLiP, and returns the caption."""

    processor = AutoProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Raise an exception for non-200 status codes
        image = Image.open(response.raw)
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching image: {e}")
        return None

    text = "A picture of"
    inputs = processor(images=image, text=text, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_token=True)
    return caption

st.title("Image Captioning with BLiP by Yaqoob Alyani")
st.write("Upload an image or enter a URL to generate a caption.")

# File uploader for image input (with error handling)
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image)
        caption = generate_caption(image=image)
        if caption:
            st.success(f"Caption: {caption}")
    except Exception as e:
        st.error(f"Error processing image: {e}")

# URL input for image fetching (with error handling)
image_url = st.text_input("Enter image URL (optional)")
if image_url:
    caption = generate_caption(image_url=image_url)
    if caption:
        st.success(f"Caption: {caption}")
    else:
        st.warning("Image not found or caption generation failed.")

# Optional: Model selection dropdown
# st.selectbox("Select Model", ["Salesforce/blip-image-captioning-base", "Another model name"])

# Display any potential errors or warnings from generate_caption()
