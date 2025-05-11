import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import streamlit as st
import torch  # Add explicit torch import

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    # Use device_map="auto" to properly handle meta tensors
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", 
        device_map="auto",
        torch_dtype=torch.float32  # Explicitly use float32 to avoid precision issues
    )
    return processor, model

st.title("🖼️ Img_captions")

# Try loading the model with exception handling
try:
    processor, model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

option = st.radio("Choose input method:", ("Upload an image", "Paste image URL"))

image = None
if option == "Upload an image":
    uploaded_img = st.file_uploader("Upload an image.", type=['png', 'jpg', 'jpeg'])
    if uploaded_img:
        try:
            image = Image.open(uploaded_img).convert('RGB')
        except Exception as e:
            st.error(f"Failed to process uploaded image: {e}")

elif option == "Paste image URL":
    img_url = st.text_input("Enter an Image URL.")
    if img_url:
        try:
            response = requests.get(img_url, stream=True)
            response.raise_for_status()  # Will raise an exception for 4XX/5XX responses
            image = Image.open(response.raw).convert('RGB')
        except Exception as e:
            st.error(f"Failed to load image from URL: {e}")

if image:
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Generating caption..."):
        try:
            inputs = processor(image, return_tensors="pt")
            
            # Handle device placement - let the model's device_map handle it
            # No need to explicitly move inputs with device_map="auto"
            
            out = model.generate(**inputs)
            txt_out = processor.decode(out[0], skip_special_tokens=True)
            
            st.subheader("📝 Caption:")
            st.success(txt_out)
        except Exception as e:
            st.error(f"Error generating caption: {e}")