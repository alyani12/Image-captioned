import streamlit as st
from PIL import Image
import requests
import torch
from transformers import AutoProcessor, BlipForConditionalGeneration

# Function to generate captions
def generate_caption(image=None, image_url=None, model_name="Salesforce/blip-image-captioning-base"):
    """Generates a caption using BLiP for a given image or image URL."""
    processor = AutoProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    # Fetch and process image
    if image_url:
        try:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()  # Raise an exception for non-200 status codes
            image = Image.open(response.raw)
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching image: {e}")
            return None
    elif image is None:
        st.error("No image or image URL provided.")
        return None

    # Generate caption
    text = "A picture of"
    inputs = processor(images=image, text=text, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Streamlit app
st.set_page_config(page_title="Image Captioning with BLiP", layout="wide")

# Add a background image to the app
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://images.pexels.com/photos/5685274/pexels-photo-5685274.jpeg');
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Image Captioning with BLiP by Yaqoob Khan ALyani")
st.write("Upload an image or enter a URL to generate a caption.")

# File uploader for image input
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image)
        caption = generate_caption(image=image)
        if caption:
            st.success(f"Caption: {caption}")
    except Exception as e:
        st.error(f"Error processing image: {e}")

# URL input for image fetching
image_url = st.text_input("Enter image URL (optional)")
if image_url:
    caption = generate_caption(image_url=image_url)
    if caption:
        st.success(f"Caption: {caption}")
    else:
        st.warning("Image not found or caption generation failed.")
